package com.johnsnowlabs.nlp.pretrained

import com.johnsnowlabs.nlp.annotator.WordEmbeddingsModel
import com.johnsnowlabs.nlp.annotators.ner.crf.NerCrfModel
import com.johnsnowlabs.nlp.annotators.pos.perceptron.PerceptronModel
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.util.Version
import org.apache.spark.ml.{PipelineModel, PipelineModelCompat}
import org.scalatest.{BeforeAndAfterEach, FlatSpec}

import java.sql.Timestamp


class ResourceDownloaderSpec extends FlatSpec with BeforeAndAfterEach {
  val b = CloudTestResources

  override def beforeEach(): Unit = {
    super.beforeEach()
    ResourceHelper.spark
  }

  "CloudResourceMetadata" should "serialize and deserialize correctly" in {
    val resource = new ResourceMetadata("name",
      Some("en"),
      Some(Version(1,2,3)),
      Some(Version(5,4,3)),
      true,
      new Timestamp(123213))

    val json = ResourceMetadata.toJson(resource)
    val deserialized = ResourceMetadata.parseJson(json)

    assert(deserialized == resource)
  }

  "CloudResourceDownloader" should "choose the newest versions" in {
    val found = ResourceMetadata.resolveResource(b.all, ResourceRequest("name", Some("en"), "", Version(1, 2, 3), Version(3, 4, 5)))

    assert(found.isDefined)
    assert(found.get == b.name_en_123_345_new)
  }

  "CloudResourceDownloader" should "filter disabled resources" in {
    val found = ResourceMetadata.resolveResource(List(b.name_en_new_disabled), ResourceRequest("name", Some("en"), "", Version(1, 2, 3), Version(3, 4, 5)))

    assert(found.isEmpty)
  }

  "CloudResourceDownloader" should "filter language and allow empty versions" in {
    val found = ResourceMetadata.resolveResource(List(b.name_en_old, b.name_de), ResourceRequest("name", Some("en"), "", Version(1, 2, 3), Version(3, 4, 5)))

    assert(found.isDefined)
    assert(found.get == b.name_en_old)
  }

  "ResourceDownloader" should "list all available public models" in {
    for (lang <- Seq("en", "de", "fr", "it")) {
      ResourceDownloader.showPublicModels(lang)
    }
  }

  "ResourceDownloader" should "list all available public pipelines" in {
    for (lang <- Seq("en", "de", "fr", "it")) {
      ResourceDownloader.showPublicPipelines(lang)
    }
  }

  lazy val standardDir: String = System.getProperty("user.home") + "/spark-nlp-models"
  lazy val compatDir: String = System.getProperty("user.home") + "/spark-nlp-models-compat"
  lazy val spark3Dir: String = System.getProperty("user.home") + "spark-nlp-models-spark3"
  lazy val langNameMap = Map(
    "en" -> "pos_anc",
    "de" -> "pos_ud_hdt",
    "fr" -> "pos_ud_gsd",
    "it" -> "pos_ud_isdt"
  )

  "PerceptronModel" should "Download & Save Standard" in {
    for ((lang, name) <- langNameMap) {
      PerceptronModel
        .pretrained(name = name, lang = lang)
        .write.overwrite()
        .save(s"$standardDir/${name}_$lang")
    }
  }

  "PerceptronModel" should "Load Standard & Save Compat" in {
    for ((lang, name) <- langNameMap) {
      PerceptronModel
        .load(s"$standardDir/${name}_$lang")
        .compatWrite.overwrite()
        .save(s"$compatDir/${name}_$lang")
    }
  }

  "PerceptronModel" should "Load Compat" in {
    for ((lang, name) <- langNameMap) {
      PerceptronModel
        .compatRead
        .load(s"$compatDir/${name}_$lang")
    }
  }

  "WordEmbeddingsModel" should "Download & Save Standard" in {
    WordEmbeddingsModel
      .pretrained(name = "glove_100d", lang = "en")
//      .write.overwrite()
//      .save(s"$standardDir/glove_100d_en")
  }

  "WordEmbeddingsModel" should "Load Standard & Save Compat" in {
    WordEmbeddingsModel
      .load(s"$standardDir/glove_100d_en")
      .compatWrite.overwrite()
      .save(s"$compatDir/glove_100d_en")
  }

  "WordEmbeddingsModel" should "Load Compat" in {
    WordEmbeddingsModel
      .compatRead
      .load(s"$compatDir/glove_100d_en")
  }

  "NerCrfModel" should "Download & Save Standard" in {
    NerCrfModel
      .pretrained(name = "ner_crf", lang = "en")
//      .write.overwrite()
//      .save(s"$standardDir/ner_crf_en")
  }

  "NerCrfModel" should "Load Standard & Save Compat" in {
    NerCrfModel
      .load(s"$standardDir/ner_crf_en")
      .compatWrite.overwrite()
      .save(s"$compatDir/ner_crf_en")
  }

  "NerCrfModel" should "Load Compat" in {
    NerCrfModel
      .compatRead
      .load(s"$compatDir/ner_crf_en")
  }

  "PretrainedPipeline" should "Download & Save Standard" in {
    val pipeline = PretrainedPipeline("explain_document_ml")
    println(s"model: ${pipeline.model.getClass.getCanonicalName}, ${pipeline.model.stages.length} stages")
    pipeline.model.stages.foreach { transformer =>
      println(transformer.getClass.getCanonicalName)
    }

    pipeline.model
      .write.overwrite()
      .save(s"$standardDir/explain_document_ml_en")
  }

  "PretrainedPipeline" should "Load Standard & Save Compat" in {
    val model = PipelineModel
      .load(s"$standardDir/explain_document_ml_en")
    println(s"model: ${model.getClass.getCanonicalName}, ${model.stages.length} stages")
    model.stages.foreach { transformer =>
      println(transformer.getClass.getCanonicalName)
    }
    PipelineModelCompat
      .fromPipelineModel(model)
      .write.overwrite()
      .save(s"$compatDir/explain_document_ml_en")
  }

  "PretrainedPipeline" should "Load Compat" in {
    println(s"*** READING PipelineModelCompat from '$compatDir/explain_document_ml_en'...")
    val compat = PipelineModelCompat
      .load(s"$compatDir/explain_document_ml_en")
    println(s"*** LOADED PipelineModelCompat(${compat.uid}) from '$compatDir/explain_document_ml_en'")
    val original = PipelineModelCompat
      .toPipelineModel(compat)
    println(s"*** CONVERTED PipelineModelCompat(${compat.uid}) to standard PipelineModel(${original.uid})")
  }
}