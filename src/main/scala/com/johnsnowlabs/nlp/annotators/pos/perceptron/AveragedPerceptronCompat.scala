package com.johnsnowlabs.nlp.annotators.pos.perceptron

import scala.collection.mutable.{Map => MMap}

import scala.collection.JavaConverters._

/**
  * Created by Saif Addin on 5/16/2017.
  */

/**
  * @param tags Holds all unique tags based on training
  * @param taggedWordBook Contains non ambiguous words and their tags
  * @param featuresWeight Contains prediction information based on context frequencies
  */

case class AveragedPerceptronCompat(
                          tags: Array[String],
                          taggedWordBook: java.util.Map[String, String],
                          featuresWeight: java.util.Map[String, Map[String, Double]]
                        ) extends Serializable {

  def predict(features: Map[String, Int]): String = {
    val weights = featuresWeight.asScala
    /**
      * scores are used for feature scores, which are all by default 0
      * if a feature has a relevant score, look for all its possible tags and their scores
      * multiply their weights per the times they appear
      * Return highest tag by score
      *
      */
    val scoresByTag = features
      .collect { case (feature, value) if weights.contains(feature) && value != 0 =>
        weights(feature).map{ case (tag, weight) =>
          (tag, value * weight)
        }
      }.aggregate(Map[String, Double]())(
      (tagsScores, tagScore) => tagScore ++ tagsScores.map{case(tag, score) => (tag, tagScore.getOrElse(tag, 0.0) + score)},
      (pTagScore, cTagScore) => pTagScore.map{case (tag, score) => (tag, cTagScore.getOrElse(tag, 0.0) + score)}
    )
    /**
      * ToDo: Watch it here. Because of missing training corpus, default values are made to make tests pass
      * Secondary sort by tag simply made to match original python behavior
      */
    tags.maxBy{ tag => (scoresByTag.withDefaultValue(0.0)(tag), tag)}
  }

  private[nlp] def getTags: Array[String] = tags
  def getWeights: Map[String, Map[String, Double]] = featuresWeight.asScala.toMap
  def getTaggedBook: Map[String, String] = taggedWordBook.asScala.toMap
}
