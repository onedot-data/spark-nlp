package com.johnsnowlabs.nlp.pretrained

import com.johnsnowlabs.nlp.annotator.PerceptronModel
import com.johnsnowlabs.nlp.annotators.pos.perceptron.PerceptronCompatModel

import java.sql.Timestamp
import com.johnsnowlabs.util.Version
import org.scalatest.FlatSpec


class ResourceDownloaderSpec extends FlatSpec {
  val b = CloudTestResources

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

  "Compat" should "Save Compat versions of POS models" in {
    val saveDir = System.getProperty("user.home") + "/compat"

    for (lang <- Seq("en", "de", "fr", "it")) {
      val original = PerceptronModel.pretrained(lang = lang)
      val compat = PerceptronCompatModel.fromPerceptronModel(original)
      compat.save(saveDir + "/pos_anc_compat_" + lang)
    }
  }

}