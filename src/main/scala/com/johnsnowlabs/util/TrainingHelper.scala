package com.johnsnowlabs.util

import com.johnsnowlabs.nlp.pretrained.ResourceType.ResourceType
import com.johnsnowlabs.nlp.pretrained.{ResourceMetadata, ResourceType}
import org.apache.commons.io.FileUtils
import org.apache.spark.ml.util.MLWriter

import java.io.File
import java.nio.file.{Files, Paths, StandardCopyOption}
import java.sql.Timestamp
import java.util.Date


object TrainingHelper {

  def saveModel(name: String,
                language: Option[String],
                libVersion: Option[Version],
                sparkVersion: Option[Version],
                modelWriter: MLWriter,
                folder: String,
                category: Option[ResourceType] = Some(ResourceType.NOT_DEFINED)
               ): Unit = {

    // 1. Get current timestamp
    val timestamp = new Timestamp(new Date().getTime)
    println(s"timestamp := ${timestamp}")


    // 2. Save model to file
    val file = Paths.get(folder, s"${name}${language.map("_" + _).getOrElse("")}-${timestamp.getTime}").toString.replaceAllLiterally("\\", "/")
    println(s"file := ${file}")
    modelWriter.save(file)

    // 3. Zip file
    val tempzipFile = Paths.get(folder, timestamp.getTime + ".zip")
    println(s"tempzipFile := ${tempzipFile}")
    ZipArchiveUtil.zip(file, tempzipFile.toString)

    // 4. Set checksum
    val checksum = FileHelper.generateChecksum(tempzipFile.toString)
    println(s"checksum := ${checksum}")

    // 5. Create resource metadata
    val meta = new ResourceMetadata(name, language, libVersion, sparkVersion, true, timestamp, true, category = category, checksum)
    println(s"meta := ${meta}")

    val zipfile = Paths.get(folder, meta.fileName)
    println(s"zipfile := ${zipfile}")


    // 6. Move the zip
    Files.move(tempzipFile, zipfile, StandardCopyOption.REPLACE_EXISTING)

    // 7. Remove original file
    try {
      FileUtils.deleteDirectory(new File(file))
    } catch {
      case _: java.io.IOException => //file lock may prevent deletion, ignore and continue
    }

    // 6. Add to metadata.json info about resource
    val metadataFile = Paths.get(folder, "metadata.json").toString
    ResourceMetadata.addMetadataToFile(metadataFile, meta)
  }
}
