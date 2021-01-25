package com.johnsnowlabs.nlp.annotators.pos.perceptron

import com.johnsnowlabs.nlp.annotators.common._
import com.johnsnowlabs.nlp.pretrained.ResourceDownloader
import com.johnsnowlabs.nlp.serialization.StructFeature
import com.johnsnowlabs.nlp.{Annotation, AnnotatorModel, ParamsAndFeaturesReadable}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.Row
import org.apache.spark.sql.types._

/**
  * Part of speech tagger that might use different approaches
  * @param uid Internal constructor requirement for serialization of params
  * @@model: representation of a POS Tagger approach
  */
class PerceptronModel(override val uid: String) extends AnnotatorModel[PerceptronModel] with PerceptronUtils {

  import com.johnsnowlabs.nlp.AnnotatorType._

  /** Internal structure for target sentences holding their range information which is used for annotation */
  private case class SentenceToBeTagged(tokenizedSentence: TokenizedSentence, start: Int, end: Int)

  val model: StructFeature[AveragedPerceptron] = {
    new StructFeature[AveragedPerceptron](this, "POS Model",
      schema = StructType(Seq(
        StructField("tags", ArrayType(StringType)),
        StructField("taggedWordBook", MapType(keyType = StringType, valueType = StringType)),
        StructField("featuresWeight", MapType(keyType = StringType, valueType = MapType(
          keyType = StringType, valueType = DoubleType
        )))
      )),
      encode = struct => {
        Row(
          struct.tags,
          struct.taggedWordBook,
          struct.featuresWeight
        )
      },
      decode = row => {
        AveragedPerceptron(
          tags = row.getAs[Seq[String]]("tags").toArray,
          taggedWordBook = row.getAs[Map[String, String]]("taggedWordBook"),
          featuresWeight = row.getAs[Map[String, Map[String, Double]]]("featuresWeight")
        )
      }
    )
  }

  override val outputAnnotatorType: AnnotatorType = POS

  override val inputAnnotatorTypes: Array[AnnotatorType] = Array(TOKEN, DOCUMENT)

  /**
    * Tags a group of sentences into POS tagged sentences
    * The logic here is to create a sentence context, run through every word and evaluate its context
    * Based on how frequent a context appears around a word, such context is given a score which is used to predict
    * Some words are marked as non ambiguous from the beginning
    * @param tokenizedSentences Sentence in the form of single word tokens
    * @return A list of sentences which have every word tagged
    */
  def tag(tokenizedSentences: Array[TokenizedSentence]): Array[TaggedSentence] = {
    //logger.debug(s"PREDICTION: Tagging:\nSENT: <<${tokenizedSentences.map(_.condense).mkString(">>\nSENT<<")}>> model weight properties in 'bias' " +
      //s"feature:\nPREDICTION: ${$$(model).getWeights("bias").mkString("\nPREDICTION: ")}")
    var prev = START(0)
    var prev2 = START(1)
    tokenizedSentences.map(sentence => {
      val context: Array[String] = START ++: sentence.tokens.map(normalized) ++: END
      sentence.indexedTokens.zipWithIndex.map { case (IndexedToken(word, begin, end), i) =>
        val tag = $$(model).getTaggedBook.getOrElse(word.toLowerCase,
          {
            val features = getFeatures(i, word, context, prev, prev2)
            $$(model).predict(features)
          }
        )
        prev2 = prev
        prev = tag
        IndexedTaggedWord(word, tag, begin, end)
      }
    }).map(TaggedSentence(_))
  }

  def this() = this(Identifiable.randomUID("POS"))

  def getModel: AveragedPerceptron = $$(model)

  def setModel(targetModel: AveragedPerceptron): this.type = set(model, targetModel)

  /** One to one annotation standing from the Tokens perspective, to give each word a corresponding Tag */
  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {
    val tokenizedSentences = TokenizedWithSentence.unpack(annotations)
    val tagged = tag(tokenizedSentences.toArray)
    PosTagged.pack(tagged)
  }
}

trait PretrainedPerceptronModel {
  def pretrained(name: String = "pos_anc", lang: String = "en", remoteLoc: String = ResourceDownloader.publicLoc): PerceptronModel = {
    ResourceDownloader.downloadModel(PerceptronModel, name, Option(lang), remoteLoc)
  }
}

object PerceptronModel extends ParamsAndFeaturesReadable[PerceptronModel] with PretrainedPerceptronModel