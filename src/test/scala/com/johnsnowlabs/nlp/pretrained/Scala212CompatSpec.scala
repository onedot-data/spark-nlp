package com.johnsnowlabs.nlp.pretrained

import com.johnsnowlabs.nlp.annotator.{NerCrfModel, PerceptronModel, WordEmbeddingsModel}
import com.johnsnowlabs.nlp.pretrained.ResourceDownloader._
import com.johnsnowlabs.nlp.pretrained.ResourceType.{MODEL, PIPELINE, ResourceType}
import com.johnsnowlabs.nlp.pretrained.Scala212CompatSpec.ExportDefinition
import com.johnsnowlabs.nlp.pretrained.Scala212CompatSpec.exports._
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.util.TrainingHelper.saveModel
import org.apache.commons.io.FileUtils.deleteDirectory
import org.apache.spark.ml.util.{MLReader, MLWriter}
import org.apache.spark.ml.{PipelineModel, PipelineModelCompat}
import org.scalatest.{BeforeAndAfterAll, BeforeAndAfterEach, FunSpec}

import java.io.File

/**
  *  This should only be run with spark-nlp compiled with scala-2.11!
  *
  *  When compiled with scala-2.12, you can comment out the
  *     it("Download & Save Standard + Compat")
  *     it("Load Standard & Save Compat")
  *  And test that the compat model can be read with scala-2.12 correctly, in:
  *     it("Load Compat & Export zip")
  **/
class Scala212CompatSpec extends FunSpec with BeforeAndAfterAll with BeforeAndAfterEach {

  // Standard format models directory. It uses java serialization, not compatible across scala versions.
  lazy val standardDir: File = new File(System.getProperty("user.home") + "/spark-nlp-models/standard")
  // Compat format models directory. It uses structured parquet DataFrames. Compatible across scala versions.
  lazy val compatDir: File = new File(System.getProperty("user.home") + "/spark-nlp-models/compat")
  // Export directory for zipped compat models using the standardized filename.
  // This is the directory that should be uploaded to S3. Client code that wish to use pre-trained models with
  // scala-2.12 should point to it instead of the default public location.
  lazy val exportDir: File = new File(System.getProperty("user.home") + "/spark-nlp-models/export")

  override def beforeAll(): Unit = {
    super.beforeAll()
    deleteDirectory(standardDir)
    deleteDirectory(compatDir)
    deleteDirectory(exportDir)
    standardDir.mkdirs()
    compatDir.mkdirs()
    exportDir.mkdirs()
  }

  override def beforeEach(): Unit = {
    super.beforeEach()
    ResourceHelper.spark
  }

  downloadConvertAndExport("Perceptron", perceptronModel)

  downloadConvertAndExport("WordEmbeddings", wordEmbeddingsModel)

  downloadConvertAndExport("NerCrf", nerCrfModel)

  downloadConvertAndExport("PretrainedPipeline", pretrainedPipeline)

  private[this] def downloadConvertAndExport[A](modelName: String, exportDef: ExportDefinition[A]): Unit = {
    describe(modelName) {
      for ((lang, name) <- exportDef.models) yield {
        describe(s"${name}_$lang") {
          val fileName = s"${name}_$lang"
          it("Download & Save Standard + Compat") {
            val model = exportDef.download(name, Some(lang))
            exportDef.standardWriter(model).save(s"$standardDir/$fileName")
          }
          it("Load Standard & Save Compat") {
            val model = exportDef.standardReader.load(s"$standardDir/$fileName")
            exportDef.compatWriter(model).save(s"$compatDir/$fileName")
          }
          it("Load Compat & Export zip") {
            val model = exportDef.compatReader.load(s"$compatDir/$fileName")
            // saveModel will:
            // - append version and timestamp to fhe filename
            // - zip the model
            // - update the manifest metadata file
            saveModel(
              name = name,
              language = Some(lang),
              libVersion = Some(libVersion),
              sparkVersion = Some(sparkVersion),
              modelWriter = exportDef.compatWriter(model),
              folder = exportDir.toString,
              category = Some(exportDef.resourceType)
            )
          }
        }
      }
    }
  }
}

object Scala212CompatSpec {
  case class ExportDefinition[A](resourceType: ResourceType,
                                 download: (String, Option[String]) => A,
                                 standardReader: MLReader[A],
                                 standardWriter: A => MLWriter,
                                 compatReader: MLReader[A],
                                 compatWriter: A => MLWriter,
                                 models: Seq[(String, String)])

  object exports {
    val perceptronModel: ExportDefinition[PerceptronModel] = {
      ExportDefinition[PerceptronModel](
        MODEL,
        downloadModel[PerceptronModel](PerceptronModel, _, _, publicLoc),
        PerceptronModel.read,
        _.write,
        PerceptronModel.compatRead,
        _.compatWrite,
        Seq(
          "en" -> "pos_anc",
          "de" -> "pos_ud_hdt",
          "fr" -> "pos_ud_gsd",
          "it" -> "pos_ud_isdt"
        )
      )
    }
    val wordEmbeddingsModel: ExportDefinition[WordEmbeddingsModel] = {
      ExportDefinition[WordEmbeddingsModel](
        MODEL,
        downloadModel[WordEmbeddingsModel](WordEmbeddingsModel, _, _, publicLoc),
        WordEmbeddingsModel.read,
        _.write,
        WordEmbeddingsModel.compatRead,
        _.compatWrite,
        Seq("en" -> "glove_100d")
      )
    }
    val nerCrfModel: ExportDefinition[NerCrfModel] = {
      ExportDefinition[NerCrfModel](
        MODEL,
        downloadModel[NerCrfModel](NerCrfModel, _, _, publicLoc),
        NerCrfModel.read,
        _.write,
        NerCrfModel.compatRead,
        _.compatWrite,
        Seq("en" -> "ner_crf")
      )
    }
    val pretrainedPipeline: ExportDefinition[PipelineModel] = {
      ExportDefinition[PipelineModel](
        PIPELINE,
        downloadPipeline(_, _, publicLoc),
        PipelineModel.read,
        _.write,
        (path: String) => PipelineModelCompat.toPipelineModel(PipelineModelCompat.load(path)),
        PipelineModelCompat.fromPipelineModel(_).compatWrite,
        Seq("en" -> "explain_document_ml")
      )
    }
  }
}
