package com.johnsnowlabs.nlp

import org.apache.spark.ml.util.{DefaultParamsReadable, MLReader}
import org.apache.spark.sql.SparkSession
import org.slf4j.LoggerFactory

import scala.collection.mutable.ArrayBuffer

class FeaturesReader[T <: HasFeatures](baseReader: MLReader[T], onRead: (T, String, SparkSession) => Unit, readMode: Option[String] = None) extends MLReader[T] {

  private val logger = LoggerFactory.getLogger(getClass)

  override def load(path: String): T = {
    logger.debug(s"load($path) [${baseReader.getClass.getCanonicalName}]")

    val instance = baseReader.load(path)

    logger.debug(s"load($path): instance ${instance.getClass.getCanonicalName}")

    for (feature <- instance.features) {
      logger.debug(s"load($path): reading feature '${feature.name}'")
      val value = feature.deserialize(sparkSession, path, feature.name, readMode)
      value.foreach { definedValue =>
        logger.debug(s"load($path): ${feature.name} <- ${definedValue.getClass.getCanonicalName}")
      }
      feature.setValue(value)
    }

    onRead(instance, path, sparkSession)

    instance
  }
}

trait ParamsAndFeaturesReadable[T <: HasFeatures] extends DefaultParamsReadable[T] {

  private val readers = ArrayBuffer.empty[(T, String, SparkSession) => Unit]

  private def onRead(instance: T, path: String, session: SparkSession): Unit = {
    for (reader <- readers) {
      reader(instance, path, session)
    }
  }

  def addReader(reader: (T, String, SparkSession) => Unit): Unit = {
    readers.append(reader)
  }

  def compatRead: MLReader[T] = new FeaturesReader(
    super.read,
    (instance: T, path: String, spark: SparkSession) => onRead(instance, path, spark),
    readMode = Some("compat")
  )

  override def read: MLReader[T] = new FeaturesReader(
    super.read,
    (instance: T, path: String, spark: SparkSession) => onRead(instance, path, spark)
  )
}
