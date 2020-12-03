
val is_gpu = System.getProperty("is_gpu","false")
val is_spark23 = System.getProperty("is_spark23","false")

val spark23Ver = "2.3.4"
val spark24Ver = "2.4.6"
val sparkVer = if (is_spark23=="false") spark24Ver else spark23Ver
val scalaVer = "2.12.12"
val scalaTestVersion = "3.0.8"

/** Package attributes */

if (is_gpu.equals("true") && is_spark23.equals("true")){
  name := "spark-nlp-gpu-spark23"
}else if (is_gpu.equals("true") && is_spark23.equals("false")){
  name := "spark-nlp-gpu"
}else if (is_gpu.equals("false") && is_spark23.equals("true")){
  name := "spark-nlp-spark23"
}else{
  name := "spark-nlp"
}


organization:= "com.johnsnowlabs.nlp"

// Note: version is used to build pretrained download url.
// since the format is incompatible, this does not matter much.
// If we manage to support the old binary format, switching the version to a clean "2.2.2"
// would build the correct url.
version := "2.2.2-onedot1"

scalaVersion in ThisBuild := scalaVer

/** Spark-Package attributes */
//spName in ThisBuild := "JohnSnowLabs/spark-nlp"

//sparkComponents in ThisBuild ++= Seq("mllib")

licenses  += "Apache-2.0" -> url("http://opensource.org/licenses/Apache-2.0")

//spIncludeMaven in ThisBuild:= false

//spAppendScalaVersion := false

resolvers in ThisBuild += "Maven Central" at "https://central.maven.org/maven2/"

resolvers in ThisBuild += "Spring Plugins" at "https://repo.spring.io/plugins-release/"

assemblyOption in assembly := (assemblyOption in assembly).value.copy(
  includeScala = false
)

credentials += Credentials(Path.userHome / ".ivy2" / ".sbtcredentials")

//ivyScala := ivyScala.value map {
//  _.copy(overrideScalaVersion = true)
//}

/** Bintray settings */
//bintrayPackageLabels := Seq("nlp", "nlu",
//  "natural-language-processing", "natural-language-understanding",
//  "spark", "spark-ml", "pyspark", "machine-learning",
//  "named-entity-recognition", "sentiment-analysis", "lemmatizer", "spell-checker",
//  "tokenizer", "stemmer", "part-of-speech-tagger", "annotation-framework")
//
//bintrayRepository := "spark-nlp"
//
//bintrayOrganization:= Some("johnsnowlabs")

//sonatypeProfileName := "com.johnsnowlabs"

publishTo := Some(
  if (isSnapshot.value)
    Opts.resolver.sonatypeSnapshots
  else
    Opts.resolver.sonatypeStaging
)

homepage:= Some(url("https://nlp.johnsnowlabs.com"))

scmInfo:= Some(
  ScmInfo(
    url("https://github.com/JohnSnowLabs/spark-nlp"),
    "scm:git@github.com:JohnSnowLabs/spark-nlp.git"
  )
)

developers in ThisBuild:= List(
  Developer(id="saifjsl", name="Saif Addin", email="saif@johnsnowlabs.com", url=url("https://github.com/saifjsl")),
  Developer(id="maziyarpanahi", name="Maziyar Panahi", email="maziyar@johnsnowlabs.com", url=url("https://github.com/maziyarpanahi")),
  Developer(id="albertoandreottiATgmail", name="Alberto Andreotti", email="alberto@pacific.ai", url=url("https://github.com/albertoandreottiATgmail")),
  Developer(id="danilojsl", name="Danilo Burbano", email="danilo@johnsnowlabs.com", url=url("https://github.com/danilojsl")),
  Developer(id="rohit13k", name="Rohit Kumar", email="rohit@johnsnowlabs.com", url=url("https://github.com/rohit13k")),
  Developer(id="aleksei-ai", name="Aleksei Alekseev", email="aleksei@pacific.ai", url=url("https://github.com/aleksei-ai")),
  Developer(id="showy", name="Eduardo Muñoz", email="eduardo@johnsnowlabs.com", url=url("https://github.com/showy"))
)


scalacOptions in (Compile, doc) ++= Seq(
  "-doc-title",
  "Spark NLP " + version.value + " ScalaDoc"
)
target in Compile in doc := baseDirectory.value / "docs/api"

lazy val analyticsDependencies =
  if (is_spark23=="false"){
    Seq(
      "org.apache.spark" %% "spark-core" % sparkVer % "provided",
      "org.apache.spark" %% "spark-mllib" % sparkVer % "provided"
    )
  } else {
    Seq(
      "org.apache.spark" %% "spark-core" % spark23Ver % "provided",
      "org.apache.spark" %% "spark-mllib" % spark23Ver % "provided"
    )
  }

lazy val testDependencies = Seq(
  "org.scalatest" %% "scalatest" % scalaTestVersion % "test"
)

val awsJavaSdkVersion = "1.11.828"
val hadoopVersion = "2.10.0"

lazy val utilDependencies = Seq(
  "com.amazonaws" % "aws-java-sdk-s3" % awsJavaSdkVersion % Provided,
  "com.amazonaws" % "aws-java-sdk-kms" % awsJavaSdkVersion % Provided,
  "com.amazonaws" % "aws-java-sdk-core" % awsJavaSdkVersion % Provided,
  "com.typesafe" % "config" % "1.3.0",
  "org.apache.ivy" % "ivy" % "2.4.0",
  "org.rocksdb" % "rocksdbjni" % "6.5.3",
  "org.apache.hadoop" % "hadoop-aws" %  hadoopVersion % Provided,
  "com.amazonaws" % "aws-java-sdk-core" % awsJavaSdkVersion % Provided,
  "com.amazonaws" % "aws-java-sdk-s3" % awsJavaSdkVersion % Provided,
  "com.github.universal-automata" % "liblevenshtein" % "3.0.0"
    exclude("com.google.guava", "guava")
    exclude("org.apache.commons", "commons-lang3"),
  "com.navigamez" % "greex" % "1.0",
  "org.json4s" %% "json4s-ext" % "3.5.3"
)


lazy val typedDependencyParserDependencies = Seq(
  "net.sf.trove4j" % "trove4j" % "3.0.3",
  "junit" % "junit" % "4.10" % Test
)
val tensorflowDependencies: Seq[sbt.ModuleID] =
  if (is_gpu.equals("true"))
    Seq(
      "org.tensorflow" % "libtensorflow" % "1.15.0",
      "org.tensorflow" % "libtensorflow_jni_gpu" % "1.15.0"
    )
  else
    Seq(
      "org.tensorflow" % "tensorflow" % "1.15.0"
    )

lazy val root = (project in file("."))
  .settings(
    libraryDependencies ++=
      analyticsDependencies ++
        testDependencies ++
        utilDependencies ++
        tensorflowDependencies++
        typedDependencyParserDependencies
  )

assemblyMergeStrategy in assembly := {
  case PathList("apache.commons.lang3", _ @ _*)  => MergeStrategy.discard
  case PathList("org.apache.hadoop", xs @ _*)  => MergeStrategy.first
  case PathList("com.amazonaws", xs @ _*)  => MergeStrategy.last
  case PathList("com.fasterxml.jackson") => MergeStrategy.first
  case PathList("META-INF", "io.netty.versions.properties")  => MergeStrategy.first
  case PathList("org", "tensorflow", _ @ _*)  => MergeStrategy.first
  case x =>
    val oldStrategy = (assemblyMergeStrategy in assembly).value
    oldStrategy(x)
}

parallelExecution in Test := false

logBuffered in Test := false

scalacOptions ++= Seq(
  "-deprecation",                      // Emit warning and location for usages of deprecated APIs.
  "-encoding", "utf-8",                // Specify character encoding used by source files.
  "-explaintypes",                     // Explain type errors in more detail.
  "-feature",                          // Emit warning and location for usages of features that should be imported explicitly.
  "-language:implicitConversions",     // Allow definition of implicit functions called views
  "-unchecked",                        // Enable additional warnings where generated code depends on assumptions.
//  "-Xcheckinit",                       // Wrap field accessors to throw an exception on uninitialized access.
  "-Xlint:type-parameter-shadow",      // A local type parameter shadows a type already in scope.
  "-Ywarn-dead-code",                  // Warn when dead code is identified.
  "-Ywarn-unused:implicits",           // Warn if an implicit parameter is unused.
  "-Ywarn-unused:imports",             // Warn if an import selector is not referenced.
  "-Ywarn-unused:locals",              // Warn if a local definition is unused.
  "-Ywarn-unused:params",              // Warn if a value parameter is unused.
  "-Ywarn-unused:patvars",             // Warn if a variable bound in a pattern is unused.
  "-Ywarn-unused:privates",            // Warn if a private member is unused.
  "-Ywarn-value-discard"               // Warn when non-Unit expression results are unused.
)
scalacOptions in(Compile, doc) ++= Seq(
  "-groups"
)
/** Enable for debugging */
testOptions in Test += Tests.Argument("-oF")

/** Disables tests in assembly */
test in assembly := {}

/** Publish test artificat **/
publishArtifact in Test := false

/** Copies the assembled jar to the pyspark/lib dir **/
lazy val copyAssembledJar = taskKey[Unit]("Copy assembled jar to pyspark/lib")
lazy val copyAssembledJarForPyPi = taskKey[Unit]("Copy assembled jar to pyspark/sparknlp/lib")

copyAssembledJar := {
  val jarFilePath = (assemblyOutputPath in assembly).value
  val newJarFilePath = baseDirectory( _ / "python" / "lib" /  "sparknlp.jar").value
  IO.copyFile(jarFilePath, newJarFilePath)
  println(s"[info] $jarFilePath copied to $newJarFilePath ")
}

copyAssembledJarForPyPi := {
  val jarFilePath = (assemblyOutputPath in assembly).value
  val newJarFilePath = baseDirectory( _ / "python" / "sparknlp" / "lib"  / "sparknlp.jar").value
  IO.copyFile(jarFilePath, newJarFilePath)
  println(s"[info] $jarFilePath copied to $newJarFilePath ")
}
