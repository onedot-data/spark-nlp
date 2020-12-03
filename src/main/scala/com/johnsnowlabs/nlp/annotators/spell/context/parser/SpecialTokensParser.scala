package com.johnsnowlabs.nlp.annotators.spell.context.parser

import com.github.liblevenshtein.transducer.factory.TransducerBuilder
import com.github.liblevenshtein.transducer.{Algorithm, Candidate, ITransducer}
import com.johnsnowlabs.nlp.annotators.spell.context.WeightedLevenshtein
import com.navigamez.greex.GreexGenerator

import scala.collection.JavaConverters.{iterableAsScalaIterableConverter, seqAsJavaListConverter}


trait SpecialClassParser {

  val label:String

  var transducer : ITransducer[Candidate]

  val maxDist: Int

  def generateTransducer: ITransducer[Candidate]

  def replaceWithLabel(tmp: String): String = {
    if(transducer.transduce(tmp, 0).asScala.toList.isEmpty)
      tmp
    else
      label
  }

  def setTransducer(t: ITransducer[Candidate]) = {
    transducer = t
    this
  }
}

trait RegexParser extends SpecialClassParser {

  val regex:String

  override def generateTransducer: ITransducer[Candidate] = {
    // first step, enumerate the regular language
    val generator = new GreexGenerator(regex)
    val matches = generator.generateAll

    // second step, create the transducer
    new TransducerBuilder().
      dictionary(matches.asScala.toList.sorted.asJava, true).
      algorithm(Algorithm.STANDARD).
      defaultMaxDistance(maxDist).
      includeDistance(true).
      build[Candidate]
  }

}

trait VocabParser extends SpecialClassParser {

  var vocab: Set[String]

  def generateTransducer: ITransducer[Candidate] = {
    // second step, create the transducer
    new TransducerBuilder().
      dictionary(vocab.toList.sorted.asJava, true).
      algorithm(Algorithm.STANDARD).
      defaultMaxDistance(maxDist).
      includeDistance(true).
      build[Candidate]
  }

  def loadCSV(path:String, col:Option[String] = None) = {
    scala.io.Source.fromFile(path).getLines.toSet
  }
}


object DateToken extends RegexParser with WeightedLevenshtein with Serializable {

  override val regex = "(01|02|03|04|05|06|07|08|09|10|11|12)\\/([0-2][0-9]|30|31)\\/(19|20)[0-9]{2}|[0-9]{2}\\/(19|20)[0-9]{2}|[0-2][0-9]:[0-5][0-9]"
  override var transducer: ITransducer[Candidate] = generateTransducer
  override val label = "_DATE_"
  override val maxDist: Int = 2

  val dateRegex = "(01|02|03|04|05|06|07|08|09|10|11|12)/[0-3][0-9]/(1|2)[0-9]{3}".r

  def separate(word: String): String = {
    val matcher = dateRegex.pattern.matcher(word)
    if (matcher.matches) {
      word.replace(matcher.group(0), label)
    }
    else
      word
  }

  override def replaceWithLabel(tmp: String): String = separate(tmp)

}

object NumberToken extends RegexParser with Serializable {

  /* used during candidate generation(correction) - must be finite */
  override val regex = "([0-9]{1,3}(\\.|,)[0-9]{1,3}|[0-9]{1,2}(\\.[0-9]{1,2})?(%)?|[0-9]{1,4})"

  override var transducer: ITransducer[Candidate] = generateTransducer

  override val label = "_NUM_"

  override val maxDist: Int = 2

  /* used to parse corpus - potentially infite */
  private val numRegex =
    """(\-|#|\$)?([0-9]+\.[0-9]+\-[0-9]+\.[0-9]+|[0-9]+/[0-9]+|[0-9]+\-[0-9]+|[0-9]+\.[0-9]+|[0-9]+,[0-9]+|[0-9]+\-[0-9]+\-[0-9]+|[0-9]+)""".r

  def separate(word: String): String = {
    val matcher = numRegex.pattern.matcher(word)
    if(matcher.matches) {
      val result = word.replace(matcher.group(0), label)
      result
    }
    else
      word
  }

  override def replaceWithLabel(tmp: String): String = separate(tmp)

}

class MedicationClass extends VocabParser with Serializable {

  @transient
  override var vocab = Set.empty[String]
  override var transducer: ITransducer[Candidate] = null
  override val label: String = "_MED_"
  override val maxDist: Int = 3

  def this(path: String) = {
    this()
    vocab = loadCSV(path)
    transducer = generateTransducer
  }

}

object AgeToken extends RegexParser with Serializable {

  override val regex: String = "1?[0-9]{0,2}-(year|month|day)(s)?(-old)?"
  override var transducer: ITransducer[Candidate] = generateTransducer
  override val label: String = "_AGE_"
  override val maxDist: Int = 2

}


object UnitToken extends VocabParser with Serializable {

  override var vocab: Set[String] = Set("MG=", "MEQ=", "TAB",
    "tablet", "mmHg", "TMIN", "TMAX", "mg/dL", "MMOL/L", "mmol/l", "mEq/L", "mmol/L",
    "mg", "ml", "mL", "mcg", "mcg/", "gram", "unit", "units", "DROP", "intl", "KG", "mcg/inh")

  override var transducer: ITransducer[Candidate] = generateTransducer
  override val label: String = "_UNIT_"
  override val maxDist: Int = 3

}
