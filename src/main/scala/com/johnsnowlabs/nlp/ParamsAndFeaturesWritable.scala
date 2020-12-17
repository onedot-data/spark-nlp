package com.johnsnowlabs.nlp

import org.apache.spark.ml.param.Params
import org.apache.spark.ml.util.{DefaultParamsWritable, MLWriter}
import org.apache.spark.sql.SparkSession
import org.slf4j.LoggerFactory

class FeaturesWriter[T](annotatorWithFeatures: HasFeatures, baseWriter: MLWriter, onWritten: (String, SparkSession) => Unit, writeMode: Option[String] = None)
  extends MLWriter with HasFeatures {

  private val logger = LoggerFactory.getLogger(getClass)

  override protected def saveImpl(path: String): Unit = {
    logger.debug(s"saveImpl($path) [${baseWriter.getClass.getCanonicalName}]")
    baseWriter.save(path)

    for (feature <- annotatorWithFeatures.features) {
      if (feature.orDefault.isDefined) {
        logger.debug(s"saveImpl($path): ${feature.name} <- ${feature.getOrDefault.getClass.getCanonicalName}")
        feature.serializeInfer(sparkSession, path, feature.name, feature.getOrDefault, writeMode)
      }
    }

    onWritten(path, sparkSession)

  }
}

trait ParamsAndFeaturesWritable extends DefaultParamsWritable with Params with HasFeatures {

  protected def onWrite(path: String, spark: SparkSession): Unit = {}

  def compatWrite: MLWriter = {
    new FeaturesWriter(
      this,
      super.write,
      (path: String, spark: SparkSession) => onWrite(path, spark),
      writeMode = Some("compat")
    )
  }

  override def write: MLWriter = {
    new FeaturesWriter(
      this,
      super.write,
      (path: String, spark: SparkSession) => onWrite(path, spark)
    )
  }

}
