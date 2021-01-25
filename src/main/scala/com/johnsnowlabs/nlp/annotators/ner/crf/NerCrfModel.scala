package com.johnsnowlabs.nlp.annotators.ner.crf

import com.johnsnowlabs.ml.crf._
import com.johnsnowlabs.nlp.AnnotatorType._
import com.johnsnowlabs.nlp.annotators.common.Annotated.{NerTaggedSentence, PosTaggedSentence}
import com.johnsnowlabs.nlp.annotators.common._
import com.johnsnowlabs.nlp.pretrained.ResourceDownloader
import com.johnsnowlabs.nlp.serialization.{MapFeature, StructFeature}
import com.johnsnowlabs.nlp.{Annotation, AnnotatorModel, ParamsAndFeaturesReadable}
import org.apache.spark.ml.param.{BooleanParam, StringArrayParam}
import org.apache.spark.ml.util._
import org.apache.spark.sql.Row
import org.apache.spark.sql.catalyst.expressions.GenericRowWithSchema
import org.apache.spark.sql.types._


/*
  Named Entity Recognition model
 */

class NerCrfModel(override val uid: String) extends AnnotatorModel[NerCrfModel] {

  def this() = this(Identifiable.randomUID("NER"))

  val entities = new StringArrayParam(this, "entities", "List of Entities to recognize")
  val model: StructFeature[LinearChainCrfModel] = new StructFeature[LinearChainCrfModel](this, "crfModel",
    schema = StructType(Seq(
      StructField("weights", ArrayType(FloatType)),
      StructField("labels", ArrayType(StringType)),
      StructField("attrs", ArrayType(StructType(Seq(
        StructField("id", IntegerType),
        StructField("name", StringType),
        StructField("isNumerical", BooleanType)
      )))),
      StructField("attrFeatures", ArrayType(StructType(Seq(
        StructField("id", IntegerType),
        StructField("attrId", IntegerType),
        StructField("label", IntegerType)
      )))),
      StructField("transitions", ArrayType(StructType(Seq(
        StructField("stateFrom", IntegerType),
        StructField("stateTo", IntegerType)
      )))),
      StructField("featuresStat", ArrayType(StructType(Seq(
        StructField("frequency", IntegerType),
        StructField("sum", FloatType)
      ))))
    )),
    encode = struct => Row(
      struct.weights,
      struct.metadata.labels,
      struct.metadata.attrs,
      struct.metadata.attrFeatures,
      struct.metadata.transitions,
      struct.metadata.featuresStat
    ),
    decode = row => {
      new LinearChainCrfModel(
        row.getAs[Seq[Float]]("weights").toArray,
        new DatasetMetadata(
          labels = row.getAs[Seq[String]]("labels").toArray,
          attrs = row.getAs[Seq[GenericRowWithSchema]]("attrs")
            .map(inner => Attr(inner.getAs[Int]("id"), inner.getAs[String]("name"), inner.getAs[Boolean]("isNumerical")))
            .toArray,
          attrFeatures = row.getAs[Seq[GenericRowWithSchema]]("attrFeatures")
            .map(inner => AttrFeature(inner.getAs[Int]("id"), inner.getAs[Int]("attrId"), inner.getAs[Int]("label")))
            .toArray,
          transitions = row.getAs[Seq[GenericRowWithSchema]]("transitions")
            .map(inner => Transition(inner.getAs[Int]("stateFrom"), inner.getAs[Int]("stateTo")))
            .toArray,
          featuresStat = row.getAs[Seq[GenericRowWithSchema]]("featuresStat")
            .map(inner => AttrStat(inner.getAs[Int]("frequency"), inner.getAs[Float]("sum")))
            .toArray
        )
      )
    }
  )
  val dictionaryFeatures: MapFeature[String, String] = new MapFeature[String, String](this, "dictionaryFeatures", keyType = StringType, valueType = StringType)
  val includeConfidence = new BooleanParam(this, "includeConfidence", "whether or not to calculate prediction confidence by token, includes in metadata")

  def setModel(crf: LinearChainCrfModel): NerCrfModel = set(model, crf)
  def setDictionaryFeatures(dictFeatures: DictionaryFeatures): this.type = set(dictionaryFeatures, dictFeatures.dict)
  def setEntities(toExtract: Array[String]): NerCrfModel = set(entities, toExtract)
  def setIncludeConfidence(c: Boolean): this.type = set(includeConfidence, c)

  def getIncludeConfidence: Boolean = $(includeConfidence)

  setDefault(dictionaryFeatures, () => Map.empty[String, String])
  setDefault(includeConfidence, false)

  /**
  Predicts Named Entities in input sentences
    * @param sentences POS tagged sentences.
    * @return sentences with recognized Named Entities
    */
  def tag(sentences: Seq[(PosTaggedSentence, WordpieceEmbeddingsSentence)]): Seq[NerTaggedSentence] = {
    require(model.isSet, "model must be set before tagging")

    val crf = $$(model)

    val fg = FeatureGenerator(new DictionaryFeatures($$(dictionaryFeatures)))
    sentences.map{case (sentence, withEmbeddings) =>
      val instance = fg.generate(sentence, withEmbeddings, crf.metadata)

      lazy val confidenceValues = {
        val fb = new FbCalculator(instance.items.length, crf.metadata)
        fb.calculate(instance, $$(model).weights, 1)
        fb.alpha
      }

      val labelIds = crf.predict(instance)

      val words = sentence.indexedTaggedWords
        .zip(labelIds.labels)
        .zipWithIndex
        .flatMap{case ((word, labelId), idx) =>
          val label = crf.metadata.labels(labelId)

          val alpha = if ($(includeConfidence)) {
            Some(confidenceValues.apply(idx).max)
          } else None

          if (!isDefined(entities) || $(entities).isEmpty || $(entities).contains(label)) {
            Some(IndexedTaggedWord(word.word, label, word.begin, word.end, alpha))
          }
          else {
            None
          }
        }

      TaggedSentence(words)
    }
  }

  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {
    val sourceSentences = PosTagged.unpack(annotations)
    val withEmbeddings = WordpieceEmbeddingsSentence.unpack(annotations)
    val taggedSentences = tag(sourceSentences.zip(withEmbeddings))
    NerTagged.pack(taggedSentences)
  }

  def shrink(minW: Float): NerCrfModel = set(model, $$(model).shrink(minW))

  override val inputAnnotatorTypes = Array(DOCUMENT, TOKEN, POS, WORD_EMBEDDINGS)

  override val outputAnnotatorType: AnnotatorType = NAMED_ENTITY

}

trait PretrainedNerCrf {
  def pretrained(name: String = "ner_crf", lang: String = "en", remoteLoc: String = ResourceDownloader.publicLoc): NerCrfModel =
    ResourceDownloader.downloadModel(NerCrfModel, name, Option(lang), remoteLoc)
}

object NerCrfModel extends ParamsAndFeaturesReadable[NerCrfModel] with PretrainedNerCrf
