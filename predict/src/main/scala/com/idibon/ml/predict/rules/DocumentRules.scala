package com.idibon.ml.predict.rules

import java.util.concurrent.ConcurrentHashMap
import java.util.regex.{Matcher, Pattern}

import com.idibon.ml.alloy.Alloy.{Reader, Writer}
import com.idibon.ml.alloy.Codec
import com.idibon.ml.common.{Archivable, ArchiveLoader, Engine}
import com.idibon.ml.predict.util.SafeCharSequence
import com.idibon.ml.predict._
import com.typesafe.scalalogging.StrictLogging
import org.apache.spark.mllib.linalg.{SparseVector, Vector}
import org.json4s._
import org.json4s.native.JsonMethods._
import org.json4s.JsonDSL.WithDouble._

import scala.util.{Failure, Try}

/**
  * Class taking care of Document rule models. This could become a trait potentially...
  * @param label the name of the label these rules are for
  * @param rules a list of tuples of rule & weights.
  */
case class DocumentRules(label: String, rules: List[(String, Float)])
    extends PredictModel[Classification] with StrictLogging
    with Archivable[DocumentRules, DocumentRulesLoader] {

  // log a warning message if any of the rules has an invalid weight
  rules.filter(r => !DocumentRules.isValidWeight(r._2))
    .foreach({ case (phrase, weight) => {
      logger.warn(s"[$this] ignoring invalid weight $weight for $phrase")
    }})

  val reifiedType = classOf[DocumentRules]

  val ruleWeightMap = rules.filter(r => DocumentRules.isValidWeight(r._2)).toMap
  val rulesCache = DocumentRules.compileRules(ruleWeightMap.map(_._1))

  // log a warning message if any of the rules fail to compile
  rulesCache.filter(_._2.isFailure).foreach({ case (p, r) => {
    logger.warn(s"[$this] unable to compile $p: ${r.failed.get.getMessage}")
  }})

  /**
    * The model will use a subset of features passed in. This method
    * should return the ones used.
    * @return Vector (likely SparseVector) where indices correspond to features
    *         that were used.
    */
  override def getFeaturesUsed(): Vector = new SparseVector(1, Array(0), Array(0))

  /**
    * This method returns the metric that best represents model quality after training
    *
    * @return Double (e.g. AreaUnderROC)
    */
  override def getEvaluationMetric(): Double = ???

  /** Serializes the object within the Alloy
    *
    * Implementations are responsible for persisting any internal state
    * necessary to re-load the object (for example, feature-to-vector
    * index mappings) to the provided Alloy.Writer.
    *
    * Implementations may return a JObject of configuration data
    * to include when re-loading the object.
    *
    * @param writer destination within Alloy for any resources that
    *               must be preserved for this object to be reloadable
    * @return Some[JObject] of configuration data that must be preserved
    *         to reload the object. None if no configuration is needed
    */
  override def save(writer: Writer): Option[JObject] = {
    // render the list into a json list of maps.
    val jsonString = compact(render(rules))
    logger.debug(jsonString)
    // create output stream to write to
    val output = writer.resource(DocumentRules.RULE_RESOURCE_NAME)
    try {
      // write to the output stream via the codec.
      Codec.String.write(output, jsonString)
      Some(new JObject(List(JField("label", JString(this.label)))))
    } finally {
      // close the stream
      output.close()
    }

  }

  /**
    * The method used to predict from a FULL DOCUMENT!
    *
    * The model needs to handle "featurization" here.
    *
    * @param doc the JObject to pull from.
    * @param options Object of predict options.
    * @return
    */
  def predict(doc: Document, options: PredictOptions): Seq[Classification] = {
    // Takes $document out of the JObject and runs rules over them.
    val content: String = (doc.json \ "content").asInstanceOf[JString].s
    Seq(docPredict(content, options.includeSignificantFeatures))
  }

  /**
    * Predicts on a piece of content.
    * @param content
    * @param significantFeatures
    * @return
    */
  def docPredict(content: String, significantFeatures: Boolean): Classification = {
    val matchesCount = getDocumentMatchCounts(content)
    // calculate pseudo prob.
    val (pseudoProb, totalCount, whiteOrBlackRule) = calculatePseudoProbability(matchesCount)
    // significant features are anything that matched; with the caveat that if white/blacklist rules
    // were triggered, only they are significant
    // only take rules that have a count greater than 0 and one of:
    // - not a white or black rule matches
    // - it's 1.0 and white or black rule matches
    // - it's 0.0 and white or black rule matches
    val sigFeatures = if (significantFeatures) {
      matchesCount.filter(x =>
        x._2 > 0 &&
        (!whiteOrBlackRule ||
          (ruleWeightMap.getOrElse(x._1, -1.0) == 1.0 && whiteOrBlackRule) ||
          (ruleWeightMap.getOrElse(x._1, -1.0) == 0.0 && whiteOrBlackRule)))
        // get weights out
        .map(x => (RuleFeature(x._1) -> this.ruleWeightMap.getOrElse(x._1, -1.0f))).toList
    } else {
      List()
    }

    if (whiteOrBlackRule) {
      Classification(this.label, pseudoProb, totalCount,
        PredictResultFlag.mask(PredictResultFlag.FORCED), sigFeatures)
    } else {
      Classification(this.label, pseudoProb, totalCount,
        PredictResultFlag.NO_FLAGS, sigFeatures)
    }
  }

  /**
    * Calculates the psuedo prob value of all the rules acting on this piece of content.
    * 1.0 == whitelist
    * 0.0 == blacklist
    * Based on the JRuby code in idisage.
    *
    * @param countMap
    * @return Tuple of Probability, and MatchCount
    */
  def calculatePseudoProbability(countMap: Map[String, Int]): (Float, Int, Boolean) = {
    // prob label [sum of (weight * count)] / [sum totalRule hits]
    var weightedSum = 0.0f
    var totalCount = 0
    var whiteListCount = 0
    var blackListCount = 0
    // for each rule
    countMap.foreach {
      case (rule, count) => {
        if (count != 0) {
          val weight: Float = ruleWeightMap.getOrElse(rule, 0.0f)
          // add to numerator
          weightedSum += (count * weight)
          // add to denominator
          totalCount += count
          // whitelist & blacklist counts
          if (weight == 1.0) {
            whiteListCount += count
          } else if (weight == 0.0) {
            blackListCount += count
          }
        }
      }
    }
    if (whiteListCount > 0 || blackListCount > 0) {
      // if any black list or white list valued rules hit, average them,
      // else this will be 1.0 or 0.0
      (whiteListCount.toFloat / (blackListCount + whiteListCount).toFloat, blackListCount + whiteListCount, true)
    } else if (totalCount > 0.0) {
      (weightedSum / totalCount, totalCount, false)
    } else {
      (0.0f, totalCount, false)
    }
  }

  /**
    * Finds all the matches and returns their starting and ending indexes.
    * @param matcher the already matched pattern object.
    * @return a list of integer tuples - (startIndex, endIndex)
    */
  def getMatches(matcher: Matcher): List[(Int, Int)] = {
    var matches: List[(Int, Int)] = List[(Int, Int)]()
    while (matcher.find()) {
      val start = matcher.start()
      val end = matcher.end()
      // append to the end of the list
      matches = matches :+(start, end)
    }
    matches
  }

  /**
    * Method to get counts of rules matching.
    * For documents we only care about counts.
    * @param content the content to match rules against
    * @return immutable map of rule count matches
    */
  def getDocumentMatchCounts(content: String) = {

    // in parallel apply the following & flatten
    rulesCache.par.map({ case (rule, pat) => {
      pat.flatMap(p => {
        /* create a matcher for the content given the compiled pattern,
         * and locate all matches; catch any errors thrown (e.g., due to
         * excessive backtracking */
        Try(getMatches(p.matcher(new SafeCharSequence(content,
          SafeCharSequence.MAX_REGEX_BACKTRACKS)))) })
        // count all of the detected matches for each rule
        .map(matchList => (rule -> matchList.size))
      /* only return successful matches that are found
       * FIXME: this should return errors, so that they can be reported
       * to the caller
       */
    }}).filter(result => result.isSuccess && result.get._2 > 0)
      .map(_.get).toMap[String, Int].seq
  }
}

/** Paired loader class for DocumentRules instances */
class DocumentRulesLoader extends ArchiveLoader[DocumentRules] {
    /** Reloads the object from the Alloy
    *
    * @param reader location within Alloy for loading any resources
    *               previous preserved by a call to
    *               { @link com.idibon.ml.common.Archivable#save}
    * @param config archived configuration data returned by a previous
    *               call to { @link com.idibon.ml.common.Archivable#save}
    * @return this object
    */
  override def load(engine: Engine, reader: Option[Reader], config: Option[JObject]): DocumentRules = {
    // it was not compiling without this implicit line...  ¯\_(ツ)_/¯
    implicit val formats = org.json4s.DefaultFormats
    val label = (config.get \ "label").extract[String]
    val jsonObject: JValue = parse(
      Codec.String.read(reader.get.resource(DocumentRules.RULE_RESOURCE_NAME)))

    val ruleJsonValue = jsonObject.extract[List[Map[String, Float]]]
    val rules = ruleJsonValue.flatMap(x => x.toList)
    new DocumentRules(label, rules)
  }
}

/** Constants for DocumentRules */
private[this] object DocumentRules {
  val RULE_RESOURCE_NAME: String = "rules.json"

  /** Tests if the user-provided rule weight is valid
    *
    * @param w - weight
    * @return true if the weight is in the valid range, false otherwise
    */
  def isValidWeight(w: Float) = w >= 0.0 && w <= 1.0

  /**
    * Helper method to answer the question, whether the rule is a regular expression or not.
    * @param rule
    * @return
    */
  def isRegexRule(rule: String): Boolean = {
    rule != null && rule.startsWith("/") && rule.endsWith("/") && rule.length() > 2
  }

  /** Tries to precompile rule phrases to Pattern instances
    *
    * @param rules - rule phrases to compile
    * @return A map from the raw rule phrase to the compilation results
    */
  def compileRules(rules: Iterable[String]) = {
    rules.par.map(phrase => {
      val pattern = Try({
        if (isRegexRule(phrase))
          Pattern.compile(phrase.substring(1, phrase.length() - 1))
        else
          Pattern.compile(phrase, Pattern.LITERAL | Pattern.CASE_INSENSITIVE)
      })
      (phrase, pattern)
    }).toList
  }
}
