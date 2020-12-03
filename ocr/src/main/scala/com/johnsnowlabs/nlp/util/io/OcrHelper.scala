package com.johnsnowlabs.nlp.util.io

import java.awt.{Image, Rectangle}
import java.awt.image.{BufferedImage, DataBufferByte, RenderedImage}
import java.io._
import java.net.URI
import java.nio.file.{Files, Paths}

import javax.imageio.ImageIO
import javax.media.jai.PlanarImage
import net.sourceforge.tess4j.ITessAPI.{TessOcrEngineMode, TessPageIteratorLevel, TessPageSegMode}
import net.sourceforge.tess4j.util.LoadLibs
import org.apache.pdfbox.pdmodel.PDDocument
import org.apache.spark.sql.{Dataset, SparkSession}
import org.slf4j.LoggerFactory
import com.johnsnowlabs.nlp.util.io.schema._
import org.apache.commons.io.FileUtils
import org.apache.spark.sql.types.{IntegerType, StringType}

import scala.util.{Failure, Success, Try}


/*
 * Perform OCR/text extraction().
 * Receives a path to a set of PDFs
 * Returns one annotation for every region found on every page,
 * {result: text, metadata:{source_file: path, pagen_number: number}}
 *
 * can produce multiple annotations for each file, and for each page.
 */


object PageSegmentationMode {
  val AUTO = TessPageSegMode.PSM_AUTO
  val SINGLE_BLOCK = TessPageSegMode.PSM_SINGLE_BLOCK
  val SINGLE_WORD = TessPageSegMode.PSM_SINGLE_WORD
}

object EngineMode {
  val OEM_LSTM_ONLY = TessOcrEngineMode.OEM_LSTM_ONLY
  val DEFAULT = TessOcrEngineMode.OEM_DEFAULT
}

object PageIteratorLevel {
  val BLOCK = TessPageIteratorLevel.RIL_BLOCK
  val PARAGRAPH = TessPageIteratorLevel.RIL_PARA
  val WORD = TessPageIteratorLevel.RIL_WORD
}

object Kernels {
  val SQUARED = 0
}

object OCRMethod {
  val TEXT_LAYER = "text"
  val IMAGE_LAYER = "image"
  val IMAGE_FILE = "image_file"
}

object NoiseMethod {
  val VARIANCE = "variance"
  val RATIO = "ratio"
}

case class OcrRow(
                   text: String,
                   pagenum: Int,
                   method: String,
                   noiselevel: Double = 0.0,
                   confidence: Double = 0.0,
                   positions: Seq[PageMatrix] = null,
                   height_dimension: Double = 0.0,
                   width_dimension: Double = 0.0,
                   filename: String = ""
)

class OcrHelper extends ImageProcessing with Serializable {

  private def logger = LoggerFactory.getLogger("OcrHelper")
  private val imageFormats = Seq(".png", ".jpg")

  @transient
  private var tesseractAPI : TesseractAccess = _

  private var preferredMethod: String = OCRMethod.TEXT_LAYER
  private var fallbackMethod: Boolean = true
  private var minSizeBeforeFallback: Int = 1

  /** Tesseract exclusive settings */
  private var pageSegmentationMode: Int = TessPageSegMode.PSM_AUTO
  private var engineMode: Int = TessOcrEngineMode.OEM_LSTM_ONLY
  private var pageIteratorLevel: Int = TessPageIteratorLevel.RIL_BLOCK
  private var kernelSize:Option[Int] = None
  private var splitPages: Boolean = true
  private var splitRegions: Boolean = true
  // whether to include confidence values in the output or not
  private var useConfidence: Boolean = false

  /* if defined we resize the image multiplying both width and height by this value */
  var scalingFactor: Option[Float] = None

  /* skew correction parameters */
  private var halfAngle: Option[Double] = None
  private var resolution: Option[Double] = None

  /* whether to include noise scores or not */
  private var estimateNoise: Option[String] = None

  def setPreferredMethod(value: String): Unit = {
    require(value == OCRMethod.TEXT_LAYER || value == OCRMethod.IMAGE_LAYER, s"OCR Method must be either" +
      s"'${OCRMethod.TEXT_LAYER}' or '${OCRMethod.IMAGE_LAYER}'")
    preferredMethod = value
  }

  def getPreferredMethod: String = preferredMethod

  def setFallbackMethod(value: Boolean): Unit = {
    fallbackMethod = value
  }

  def getFallbackMethod: Boolean = fallbackMethod

  def setMinSizeBeforeFallback(value: Int): Unit = {
    minSizeBeforeFallback = value
  }

  def getMinSizeBeforeFallback: Int = minSizeBeforeFallback

  def setPageSegMode(mode: Int): Unit = {
    pageSegmentationMode = mode
  }

  def getPageSegMode: Int = {
    pageSegmentationMode
  }

  def setEngineMode(mode: Int): Unit = {
    engineMode = mode
  }

  def getEngineMode: Int = {
    engineMode
  }

  def setPageIteratorLevel(level: Int): Unit = {
    pageIteratorLevel = level
  }

  def getPageIteratorLevel: Int = {
    pageIteratorLevel
  }

  def setScalingFactor(factor:Float): Unit = {
    if (factor == 1.0f)
      scalingFactor = None
    else
      scalingFactor = Some(factor)
  }

  /* here we make sure '!pageSplit && regionSplit' cannot happen
  *  if regions are split, then you cannot merge pages(it's not possible).
  * */
  def setSplitPages(value: Boolean): Unit = {
    splitPages = value

    if(!splitPages)
      splitRegions = false
  }

  def getSplitPages: Boolean = splitPages

  def setSplitRegions(value: Boolean): Unit = {
    splitRegions = value

    if(splitRegions)
      splitPages = true
  }

  def getSplitRegions: Boolean = splitRegions

  def setIncludeConfidence(value: Boolean): Unit = {
    useConfidence = value
  }

  def getIncludeConfidence:Boolean = useConfidence

  def useErosion(useIt: Boolean, kSize:Int = 2, kernelShape:Int = Kernels.SQUARED): Unit = {
    if (!useIt)
      kernelSize = None
    else
      kernelSize = Some(kSize)
  }

  private def getListOfFiles(dir: String): List[(String, FileInputStream)] = {
    val path = new File(dir)
    if (path.exists && path.isDirectory) {
      path.listFiles.filter(_.isFile).map(f => (f.getName, new FileInputStream(f))).toList
    } else if (path.exists && path.isFile) {
      List((path.getName, new FileInputStream(path)))
    } else {
      throw new FileNotFoundException("Path does not exist or is not a valid file or directory")
    }
  }

  def createDataset(spark: SparkSession, inputPath: String): Dataset[OcrRow] = {
    import spark.implicits._
    val sc = spark.sparkContext
    val files = sc.binaryFiles(inputPath)
    files.flatMap {
      // here we handle images directly
      case (fileName, stream) if imageFormats.exists(fileName.endsWith)=>
          doImageOcr(stream.open)
            .map(_.copy(filename = fileName))

      case (fileName, stream) =>
          doPDFOcr(stream.open, fileName)
            .map(_.copy(filename = fileName))
    }.filter(_.text.nonEmpty).toDS
  }

  /* WARNING: this only makes sense with splitPages == false, otherwise the map creation discards information(complete pages)
  (multiple pages per file is not supported) */
  def createMap(inputPath: String): Map[String, String] = {
    val files = getListOfFiles(inputPath)
    files.flatMap {case (fileName, stream) =>
      doPDFOcr(stream, fileName).map(ocrrow => (fileName, ocrrow.text))
    }.filter(_._2.nonEmpty).toMap
  }

  /*
  * Enable/disable automatic skew(rotation) correction,
  *
  * @halfAngle, half the angle(in degrees) that will be considered for correction.
  * @resolution, the step size(in degrees) that will be used for generating correction
  * angle candidates.
  *
  * For example, for halfAngle = 2.0, and resolution 0.5,
  * candidates {-2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2} will be evaluated.
  * */
  def setAutomaticSkewCorrection(useIt:Boolean, halfAngle:Double = 5.0, resolution:Double = 1.0): Unit = {
    if(useIt) {
      this.halfAngle = Some(halfAngle)
      this.resolution = Some(resolution)
    } else {
      this.halfAngle = None
      this.resolution = None
    }
  }

  def setEstimateNoise(noiseMethod: String): Unit = {
    estimateNoise = Some(noiseMethod)
  }

  private def tesseract:TesseractAccess = {
    if (tesseractAPI == null)
      tesseractAPI = initTesseract()

    tesseractAPI
  }

  private def initTesseract():TesseractAccess = this.synchronized {
    val api = new TesseractAccess()
    val tessDataFolder = LoadLibs.extractTessResources("tessdata")
    api.setDatapath(tessDataFolder.getAbsolutePath)
    api.setPageSegMode(pageSegmentationMode)
    api.setOcrEngineMode(engineMode)
    api.initialize()
    api
  }

  def reScaleImage(image: PlanarImage, factor: Float): BufferedImage = {
    val width = image.getWidth * factor
    val height = image.getHeight * factor
    val scaledImg = image.getAsBufferedImage().
    getScaledInstance(width.toInt, height.toInt, Image.SCALE_AREA_AVERAGING)
    toBufferedImage(scaledImg)
  }

  /* erode the image */
  def erode(bi: BufferedImage, kernelSize: Int): BufferedImage = {
    // convert to grayscale
    val gray = new BufferedImage(bi.getWidth, bi.getHeight, BufferedImage.TYPE_BYTE_GRAY)
    val g = gray.createGraphics()
    g.drawImage(bi, 0, 0, null)
    g.dispose()

    // init
    val dest = new BufferedImage(bi.getWidth(), bi.getHeight(), BufferedImage.TYPE_BYTE_GRAY)
    val outputData = dest.getRaster.getDataBuffer.asInstanceOf[DataBufferByte].getData
    val inputData = gray.getRaster.getDataBuffer.asInstanceOf[DataBufferByte].getData

    // handle the unsigned type
    val converted = inputData.map(fromUnsigned)

    // define the boundaries of the squared kernel
    val width = bi.getWidth
    val rowIdxs = Range(-kernelSize, kernelSize + 1).map(_ * width)
    val colIdxs = Range(-kernelSize, kernelSize + 1)

    // convolution and nonlinear op (minimum)
    outputData.indices.par.foreach { idx =>
      var acc = Int.MaxValue
      for (ri <- rowIdxs; ci <- colIdxs) {
        val index = idx + ri + ci
        if (index > -1 && index < converted.length)
          if(acc > converted(index))
            acc = converted(index)
      }
      outputData(idx) = fromSigned(acc)
    }
    dest
  }

  def fromUnsigned(byte:Byte): Int = {
    if (byte > 0)
      byte
    else
      byte + 255
  }

  def fromSigned(integer:Int): Byte = {
    if (integer > 0 && integer < 127)
      integer.toByte
    else
      (integer - 255).toByte
  }


  def binarize(bi: BufferedImage): BufferedImage = {

    // convert to grayscale
    val gray = new BufferedImage(bi.getWidth, bi.getHeight, BufferedImage.TYPE_BYTE_GRAY)
    val g = gray.createGraphics()
    g.drawImage(bi, 0, 0, null)
    g.dispose()

    // init
    val dest = new BufferedImage(bi.getWidth(), bi.getHeight(), BufferedImage.TYPE_BYTE_GRAY)
    val outputData = dest.getRaster.getDataBuffer.asInstanceOf[DataBufferByte].getData
    val inputData = gray.getRaster.getDataBuffer.asInstanceOf[DataBufferByte].getData

    // handle the unsigned type
    val converted = inputData.map(fromUnsigned)

    // convolution and nonlinear op (minimum)
    // TODO must use adaptive thresholding
    outputData.indices.par.foreach { idx =>
      if (converted(idx) < 170) {
        outputData(idx) = fromSigned(2)
      }
      else
        outputData(idx) = fromSigned(250)
    }
    dest
  }

  /* compute standard deviation from histogram */
  def stdev(histogram: Array[Int]): Double = {
    val mean = histogramMean(histogram)
    val result = Math.sqrt(histogram.zipWithIndex.map{ case (x, i) => (i.toDouble - mean) * (i.toDouble - mean) * x}.sum /
      (histogram.sum.toDouble - 1))

    result
  }

  /* estimate noise score based on image histogram */
  private def computeNoiseScore(estimateNoise: Option[String], image: BufferedImage, r: Rectangle): Double =
  estimateNoise.map { method =>

    val histogram = Array.fill(256)(0)
    val imageData = image.getRaster.getDataBuffer.asInstanceOf[DataBufferByte].getData

    Range(r.x, r.x + r.width).foreach { i =>
        Range(r.y, r.y + r.height).foreach { j =>
          val pixVal = imageData(j * image.getWidth + i)
          val intVal = signedByte2UnsignedInt(pixVal)
          assert(intVal <= 255)
          assert(intVal >= 0)
          histogram(intVal) += 1
        }
    }

    if(method.equals(NoiseMethod.RATIO)) {
      /* here we should do something adaptive instead */
      val LIMIT = 10
      val denom = histogram.slice(0, LIMIT).sum + histogram.takeRight(LIMIT).sum + 1
      val num = histogram.slice(LIMIT, histogram.length - LIMIT).sum
      num.toFloat / denom
    } else { // variance
      stdev(histogram.take(128)) //+ stdev(histogram.takeRight(128))
    }
  }.getOrElse(0.0)

  private def mean(values: Seq[Double]) =
    if (values.isEmpty)
      Double.NaN
    else
      values.sum / values.size

  private def histogramMean(hist:Seq[Int]) = {
    val mass = hist.zipWithIndex.map{case (count, i) => count * i}.sum
    val count = hist.sum
    if (count == 0)
      0.0
    else
      mass.toDouble / count
  }

  // TODO: Sequence return type should be enough
  /* response here is (text, pagenum, noise_level) */
  private def tesseractMethod(renderedImages:Seq[RenderedImage]): Option[Seq[OcrRow]] = this.synchronized {
    val imageRegions = renderedImages.map(render => {
      val image = PlanarImage.wrapRenderedImage(render)

      // correct skew if parameters are provided
      val skewCorrected = halfAngle.flatMap{angle => resolution.map {res =>
        correctSkew(image.getAsBufferedImage, angle, res)
      }}.getOrElse(image.getAsBufferedImage)

      // rescale if factor provided
      val scaledImage = scalingFactor.map { factor =>
        reScaleImage(image, factor)
      }.getOrElse(skewCorrected)

      // erode if kernel provided
      val dilatedImage = kernelSize.map {kernelRadio =>
        erode(scaledImage, kernelRadio)
      }.getOrElse(scaledImage)

      // obtain regions and run OCR on each region
      val regions = {
        /** Some ugly image scenarios cause a null pointer in tesseract. Avoid here.*/
        try {
          tesseract.getSegmentedRegions(scaledImage, pageIteratorLevel).map(Some(_)).toList
        } catch {
          case _: NullPointerException =>
            logger.info(s"Tesseract failed to process a document. Falling back to text layer.")
            List()
        }
      }

      regions.flatMap(_.map { rectangle =>
        val (text, confidence) =
            tesseract.doOCR(dilatedImage, rectangle, pageIteratorLevel, useConfidence)
        (text, computeNoiseScore(estimateNoise, scaledImage, rectangle), confidence, image.getHeight, image.getWidth)
      })
    })


    (splitPages, splitRegions) match {
      case (true, true) =>
        Option(imageRegions.zipWithIndex.flatMap {case (pageRegions, pagenum) =>
          pageRegions.map{case (r, nl, conf, h, w) => OcrRow(r, pagenum, OCRMethod.IMAGE_LAYER, nl, conf, height_dimension = h.toDouble, width_dimension = w.toDouble)}})
      case (true, false) =>
        // this merges regions within each page, splits the pages
        Option(imageRegions.zipWithIndex.
              map { case (pageRegions, pagenum) =>
                val noiseLevel = mean(pageRegions.map(_._2))
                val confidence = mean(pageRegions.map(_._3))
                val mergedText = pageRegions.map(_._1).mkString(System.lineSeparator())
                val minHeight = pageRegions.map(_._4).max.toDouble
                val minWidth = pageRegions.map(_._5).max.toDouble
                OcrRow(mergedText, pagenum, OCRMethod.IMAGE_LAYER, noiseLevel, confidence, height_dimension = minHeight, width_dimension = minWidth)})
      case _ =>
        // don't split pages either regions, => everything coming from page 0
        val mergedText = imageRegions.map{pageRegions =>  pageRegions.map(_._1).
          mkString(System.lineSeparator)}.mkString(System.lineSeparator)
        // here the noise level will be an average
        val noiseLevel = mean(imageRegions.flatten.map(_._2))
        val confidence = mean(imageRegions.flatten.map(_._3))
        val minHeight = imageRegions.flatten.map(_._4).max.toDouble
        val minWidth = imageRegions.flatten.map(_._5).max.toDouble
        Option(Seq(OcrRow(mergedText, 0, OCRMethod.IMAGE_LAYER, noiseLevel, confidence, height_dimension = minHeight, width_dimension = minWidth)))
     }
  }

  private def getCoordinates(doc: PDDocument, startPage: Int, endPage: Int): Seq[PageMatrix] = {
    import scala.collection.JavaConverters._

    Range(startPage, endPage + 1).map(pagenum => {
      val stripper = new CustomStripper
      stripper.setStartPage(pagenum)
      stripper.setEndPage(pagenum)
      stripper.getText(doc)

      val line = stripper.lines.asScala.flatMap(_.textPositions.asScala)

      PageMatrix(line.toArray.map(p => {
        Mapping(
          p.toString,
          pagenum,
          p.getTextMatrix.getTranslateX,
          p.getTextMatrix.getTranslateY,
          p.getWidth,
          p.getHeightDir
        )
        })
      )
    })

  }

  private def pdfboxMethod(pdfDoc: PDDocument, startPage: Int, endPage: Int): Option[Seq[OcrRow]] = {
    if (splitPages)
      Some(Range(startPage, endPage + 1).flatMap(pagenum =>
        extractText(pdfDoc, pagenum, pagenum).map(t =>
          OcrRow(
            t,
            pagenum - 1,
            OCRMethod.TEXT_LAYER,
            positions = getCoordinates(pdfDoc, pagenum, pagenum),
            height_dimension = pdfDoc.getPage(pagenum-1).getMediaBox.getHeight,
            width_dimension = pdfDoc.getPage(pagenum-1).getMediaBox.getWidth
          )
        )
      ))
    else
      Some(extractText(pdfDoc, startPage, endPage).zipWithIndex.map{case (t, idx) =>
        OcrRow(t, idx, OCRMethod.TEXT_LAYER, positions = getCoordinates(pdfDoc, startPage, endPage),
          height_dimension = pdfDoc.getPage(startPage-1).getMediaBox.getHeight,
          width_dimension = pdfDoc.getPage(startPage-1).getMediaBox.getWidth)
      })
  }

  private def pageOcr(pdfDoc: PDDocument, startPage: Int, endPage: Int): Seq[OcrRow] = {

    val result = preferredMethod match {

      case OCRMethod.IMAGE_LAYER => tesseractMethod(getImageFromPDF(pdfDoc, startPage - 1, endPage - 1))
        .filter(_.nonEmpty)
        .filter(content => minSizeBeforeFallback == 0 || content.forall(_.text.length >= minSizeBeforeFallback))
        .orElse(if (fallbackMethod) {pdfboxMethod(pdfDoc, startPage, endPage)} else None)

      case OCRMethod.TEXT_LAYER => pdfboxMethod(pdfDoc, startPage, endPage)
        .filter(content => minSizeBeforeFallback == 0 || content.forall(_.text.length >= minSizeBeforeFallback))
        .orElse(if (fallbackMethod) {tesseractMethod(getImageFromPDF(pdfDoc, startPage - 1, endPage - 1))} else None)

      case _ => throw new IllegalArgumentException(s"Invalid OCR Method. Must be '${OCRMethod.TEXT_LAYER}' or '${OCRMethod.IMAGE_LAYER}'")
    }
    result.getOrElse(Seq.empty[OcrRow])
  }

  private def drawRectangles(doc: PDDocument, coordinates: Seq[Coordinate]): String = {

    import java.awt.Color
    import org.apache.pdfbox.pdmodel.PDPageContentStream

    val pagedCoordinates = coordinates.groupBy(_.p-1)

    pagedCoordinates.keys.toArray.sorted.foreach(pageIndex => {
      val contentStream = new PDPageContentStream(doc, doc.getPage(pageIndex), PDPageContentStream.AppendMode.APPEND, false)
      contentStream.setStrokingColor(Color.RED)

      pagedCoordinates(pageIndex).foreach(coord => {
        if (coord.p > -1) {
          contentStream.addRect(coord.x, coord.y, coord.w, coord.h)
          contentStream.stroke()
        }
      })

      contentStream.close()

    })

    val tmpFile = Files.createTempFile("sparknlp_ocr_", "").toAbsolutePath.toString
    val fileout = new File(tmpFile)
    doc.save(fileout)
    tmpFile

  }

  private def drawRectanglesToTmpSpark(sparkSession: SparkSession, inputPath: String, coordinates: Seq[Coordinate]): String = {
    val stream = sparkSession.sparkContext.binaryFiles(inputPath).first()._2
    val pdfDoc = PDDocument.load(stream.open())
    drawRectangles(pdfDoc, coordinates)
  }

  private def drawRectanglesToTmp(inputPath: String, coordinates: Seq[Coordinate]): String = {
    val target = new File(inputPath)
    require(target.exists, s"File $inputPath does not exist")
    val stream = new FileInputStream(target)
    val pdfDoc = PDDocument.load(stream)
    drawRectangles(pdfDoc, coordinates)
  }

  private def drawRectanglesToTmp(inputPath: String, coordinates: java.util.List[Coordinate]): String = {
    import scala.collection.JavaConverters._
    drawRectanglesToTmp(inputPath, coordinates.asScala)
  }

  def drawRectanglesToFile(inputPath: String, coordinates: Seq[Coordinate], outputPath: String): Unit = {
    val finalPath = drawRectanglesToTmp(inputPath, coordinates)
    FileUtils.copyFile(new File(finalPath), new File(outputPath))
  }

  def drawRectanglesToFile(inputPath: String, coordinates: java.util.List[Coordinate], outputPath: String): Unit = {
    val finalPath = drawRectanglesToTmp(inputPath, coordinates)
    FileUtils.copyFile(new File(finalPath), new File(outputPath))
  }

  def drawRectanglesDataset(
                             spark: SparkSession,
                             dataset: Dataset[_],
                             filenameCol: String = "filename",
                             pagenumCol: String = "pagenum",
                             coordinatesCol: String = "coordinates",
                             outputLocation: String = "./highlighted/",
                             outputSuffix: String = "_draw"
                           ): Unit = {

    require(dataset.columns.contains(coordinatesCol), s"Column $coordinatesCol does not exist in dataframe")
    require(dataset.select(coordinatesCol).schema.find(_.name == coordinatesCol).map(_.dataType).get == Coordinate.coordinateType, s"Column $coordinatesCol is not a valid coordinates schema type")

    require(dataset.columns.contains(filenameCol), s"Column $filenameCol does not exist in dataframe")
    require(dataset.select(filenameCol).schema.find(_.name == filenameCol).map(_.dataType).get == StringType, s"Column $filenameCol is not a string type column")

    require(dataset.columns.contains(pagenumCol), s"Column $pagenumCol does not exist in dataframe")
    require(dataset.select(pagenumCol).schema.find(_.name == pagenumCol).map(_.dataType).get == IntegerType, s"Column $pagenumCol is not an integer type column")

    import spark.implicits._

    val collection = dataset.select(filenameCol, pagenumCol, coordinatesCol).as[(String, Int, Seq[Coordinate])].collect()

    val uniquePaths = collection.map(_._1).distinct

    uniquePaths.foreach(path => {
      val coordinates = collection
        .filter(c => c._1 == path)
        .sortBy(_._2)
        .map(_._3)

      val finalPath = coordinates.foldLeft(path)((curPath, coordinate) => drawRectanglesToTmpSpark(spark, curPath, coordinate))

      val outputName = new File(new URI(path).getPath).getName
      val (name, extension) = outputName.splitAt(outputName.length-4)
      val editedName = name + outputSuffix
      FileUtils.copyFile(
        new File(finalPath),
        Paths.get(outputLocation, editedName+extension).toFile
      )
    })

  }

  /*
   * fileStream: a stream to PDF files
   * filename: name of the original file(used for failure login)
   * returns sequence of (pageNumber:Int, textRegion:String, decidedMethod:String)
   *
   * */
  private def doPDFOcr(fileStream:InputStream, filename:String):Seq[OcrRow] = {
    val pagesTry = Try(PDDocument.load(fileStream)).map { pdfDoc =>
      val numPages = pdfDoc.getNumberOfPages
      require(numPages >= 1, "pdf input stream cannot be empty")

      val result = pageOcr(pdfDoc, 1, numPages)

    /* TODO: beware PDF box may have a potential memory leak according to,
     * https://issues.apache.org/jira/browse/PDFBOX-3388
     */
      pdfDoc.close()
      result
    }

    pagesTry match {
      case Failure(e) =>
        logger.error(s"""found a problem trying to open file $filename""")
        logger.error(pagesTry.toString)
        Seq.empty
      case Success(content) =>
        content
    }
  }

  private def doImageOcr(fileStream:InputStream):Seq[OcrRow] = {
    val image = ImageIO.read(fileStream)
    tesseractMethod(Seq(image)).map(_.map(_.copy(method = OCRMethod.IMAGE_FILE))).getOrElse(Seq.empty)
  }

  /*
  * extracts a text layer from a PDF.
  * */
  private def extractText(document: PDDocument, startPage: Int, endPage: Int): Seq[String] = {
    import org.apache.pdfbox.text.PDFTextStripper
    val pdfTextStripper = new PDFTextStripper
    pdfTextStripper.setStartPage(startPage)
    pdfTextStripper.setEndPage(endPage)
    Seq(pdfTextStripper.getText(document))
  }

  private def getImageFromPDF(document: PDDocument, startPage: Int, endPage: Int): Seq[BufferedImage] = {
    Range(startPage, endPage + 1).flatMap(numPage => {
      val page = document.getPage(numPage)
      val multiImage = new MultiImagePDFPage(page)
      multiImage.getMergedImages()
    })
  }
}
