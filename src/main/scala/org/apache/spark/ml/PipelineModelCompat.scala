package org.apache.spark.ml

import com.johnsnowlabs.nlp.ParamsAndFeaturesWritable
import org.apache.hadoop.fs.Path
import org.apache.spark.SparkContext
import org.apache.spark.ml.Pipeline.SharedReadWrite
import org.apache.spark.ml.Pipeline.SharedReadWrite.getStagePath
import org.apache.spark.ml.PipelineModelCompat.PipelineModelCompatWriter
import org.apache.spark.ml.util._
import org.apache.spark.util.Utils
import org.json4s.DefaultFormats
import org.json4s.jackson.JsonMethods.{compact, parse, render}

import scala.collection.JavaConverters._
import scala.util.Try

class PipelineModelCompat private[ml](uid: String, stages: java.util.List[Transformer])
  extends PipelineModel(uid, stages) {

  def this(uid: String, stages: Seq[Transformer]) = this(uid, stages.asJava)

  override def write: MLWriter = new PipelineModelCompatWriter(this)

  def compatWrite: MLWriter = write
}

object PipelineModelCompat extends MLReadable[PipelineModelCompat] {

  def fromPipelineModel(pipelineModel: PipelineModel): PipelineModelCompat =
    new PipelineModelCompat(pipelineModel.uid, pipelineModel.stages)

  def toPipelineModel(pipelineCompatModel: PipelineModelCompat): PipelineModel =
    new PipelineModel(pipelineCompatModel.uid, pipelineCompatModel.stages)

  def compatRead: MLReader[PipelineModelCompat] = new PipelineModelCompatReader
  def read: MLReader[PipelineModelCompat] = compatRead

  private[PipelineModelCompat] class PipelineModelCompatReader extends MLReader[PipelineModelCompat] {

    /** Checked against metadata when loading model */
    private val className = classOf[PipelineModelCompat].getName

    override def load(path: String): PipelineModelCompat = {
      val (uid: String, stages: Array[PipelineStage]) = internalLoad(className, sc, path)
      val transformers = stages.map {
        case stage: Transformer => stage
        case other => throw new RuntimeException(s"PipelineModel.read loaded a stage but found it" +
          s" was not a Transformer.  Bad stage ${other.uid} of type ${other.getClass}")
      }
      new PipelineModelCompat(uid, transformers)
    }

    private def internalLoad(expectedClassName: String,
                             sc: SparkContext,
                             path: String): (String, Array[PipelineStage]) = {
      val metadata = DefaultParamsReader.loadMetadata(path, sc, expectedClassName)

      implicit val format = DefaultFormats
      val stagesDir = new Path(path, "stages").toString
      val stageUids: Array[String] = (metadata.params \ "stageUids").extract[Seq[String]].toArray
      val stages: Array[PipelineStage] = stageUids.zipWithIndex.map { case (stageUid, idx) =>
        val stagePath = SharedReadWrite.getStagePath(stageUid, idx, stageUids.length, stagesDir)
        loadParamsInstance[PipelineStage](stagePath, sc)
      }
      (metadata.uid, stages)
    }

    private def loadParamsInstance[T](path: String, sc: SparkContext): T = {
      val metadata = DefaultParamsReader.loadMetadata(path, sc)
      val cls = Utils.classForName(metadata.className)
      val method = Try(cls.getMethod("compatRead"))
        .recover { case _: NoSuchMethodException => cls.getMethod("read") }
        .get
      method.invoke(null).asInstanceOf[MLReader[T]].load(path)
    }
  }

  private[PipelineModelCompat] class PipelineModelCompatWriter(instance: PipelineModelCompat) extends MLWriter {
    import org.json4s.JsonDSL._

    SharedReadWrite.validateStages(instance.stages.asInstanceOf[Array[PipelineStage]])

    override protected def saveImpl(path: String): Unit = {
      val stageUids = instance.stages.map(_.uid)
      val jsonParams = List("stageUids" -> parse(compact(render(stageUids.toSeq))))
      DefaultParamsWriter.saveMetadata(instance, path, sc, paramMap = Some(jsonParams))

      // Save stages
      val stagesDir = new Path(path, "stages").toString
      instance.stages.zipWithIndex
        .map { case (stage, idx) =>
          val saveImpl = stage match {
            case localStage: ParamsAndFeaturesWritable =>
              localStage.compatWrite.save _
            case standardStage =>
              standardStage.asInstanceOf[MLWritable].write.save _
          }
          (stage.uid, saveImpl, idx)
        }
        .foreach { case (uid, saveImpl, idx) =>
          saveImpl(getStagePath(uid, idx, instance.stages.length, stagesDir))
        }
    }
  }
}
