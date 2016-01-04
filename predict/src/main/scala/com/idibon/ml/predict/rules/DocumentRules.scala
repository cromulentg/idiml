package com.idibon.ml.predict.rules

import java.util.concurrent.ConcurrentHashMap
import java.util.regex.{Matcher, Pattern}

import com.idibon.ml.alloy.Alloy.{Reader, Writer}
import com.idibon.ml.predict.util.{RegexInterruption, SafeCharSequence}
import com.idibon.ml.predict.{DocumentPredictionResultBuilder, DocumentPredictionResult}
import com.typesafe.scalalogging.Logger
import org.apache.spark.mllib.linalg.Vector
import org.json4s._

import scala.util.{Failure, Success, Try}

/**
  * Class taking care of Document rule models. This could become a trait potentially...
  * @param label the index of the label these rules are for
  * @param rules a list of tuples of rule & weights.
  */
class DocumentRules(label: Int, rules: List[(String, Double)]) extends RulesModel(label, rules) {

  lazy val logger = Logger(org.slf4j.LoggerFactory
    .getLogger(classOf[DocumentRules].getName))
  // Rules cache
  val rulesCache: ConcurrentHashMap[String, Try[Pattern]] = new ConcurrentHashMap[String, Try[Pattern]]()
  // Compile the rules and populate the rule cache; this could be slow, but there's no way around this...
  populateCache()

  /**
    * Helper method to compile the rules into regular expressions.
    */
  def populateCache(): Unit = {
    // do in parallel
    for ((rule, weight) <- this.rules.par) {
      if (isRegexRule(rule)) {
        val expression = Try(Pattern.compile(rule.substring(1, rule.length() - 1))) // makes it a regex
        savePatternToCache(rule, expression)
      } else {
        // makes it a regex, but we want to interpret everything literally, etc.
        val expression = Try(Pattern.compile(rule, Pattern.LITERAL | Pattern.CASE_INSENSITIVE))
        savePatternToCache(rule, expression)
      }
    }
  }

  /**
    * The method used to predict from a vector of features.
    * @param features Vector of features to use for prediction.
    * @param significantFeatures whether to return significant features.
    * @param significantThreshold if returning significant features the threshold to use.
    * @return
    */
  override def predict(features: Vector,
                       significantFeatures: Boolean,
                       significantThreshold: Double): DocumentPredictionResult = {
    throw new RuntimeException("Not implemented for rules.")
  }

  /**
    * Returns the type of model.
    * @return canonical class name.
    */
  override def getType(): String = this.getClass().getCanonicalName()

  /**
    * The model will use a subset of features passed in. This method
    * should return the ones used.
    * @return Vector (likely SparseVector) where indices correspond to features
    *         that were used.
    */
  override def getFeaturesUsed(): Vector = ???

  /** Reloads the object from the Alloy
    *
    * @param reader location within Alloy for loading any resources
    *               previous preserved by a call to
    *               { @link com.idibon.ml.feature.Archivable#save}
    * @param config archived configuration data returned by a previous
    *               call to { @link com.idibon.ml.feature.Archivable#save}
    * @return this object
    */
  override def load(reader: Reader, config: Option[JObject]): DocumentRules.this.type = ???

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
  override def save(writer: Writer): Option[JObject] = ???

  /**
    * The method used to predict from a FULL DOCUMENT!
    *
    * The model needs to handle "featurization" here.
    *
    * @param document the JObject to pull from.
    * @param significantFeatures whether to return significant features.
    * @param significantThreshold if returning significant features the threshold to use.
    * @return
    */
  override def predict(document: JObject,
                       significantFeatures: Boolean,
                       significantThreshold: Double): DocumentPredictionResult = {
    // Takes $document out of the JObject and runs rules over them.
    val content: String = (document \ "content").asInstanceOf[JString].s
    docPredict(content, significantFeatures)
  }

  /**
    * Predicts on a piece of content.
    * @param content
    * @param significantFeatures
    * @return
    */
  def docPredict(content: String, significantFeatures: Boolean): DocumentPredictionResult = {
    val dpr = new DocumentPredictionResultBuilder()
    val matchesCount: Map[String, Int] = getDocumentMatchCounts(content)
    // calculate pseudo prob.
    val (psuedoProb, totalCount) = calculatePseudoProbability(matchesCount)
    if (psuedoProb > 0.0) {
      // TODO: significant features
      dpr.addDocumentPredictResult(label, psuedoProb, List())
        .setMatchCount(totalCount)
    }
    dpr.build()
  }

  /**
    * Helper method to answer the question, whether the rule is a regular expression or not.
    * @param rule
    * @return
    */
  def isRegexRule(rule: String): Boolean = {
    rule != null && rule.startsWith("/") && rule.endsWith("/") && rule.length() > 2
  }

  /**
    * Helper method to save a compilded pattern to the cache.
    * @param rule the rule we are saving the pattern under.
    * @param expression The pattern/error we're saving.
    */
  private def savePatternToCache(rule: String, expression: Try[Pattern]): Try[Pattern] = {
    if (expression.isFailure)
      this.logger.error("Failed to compile expression [" + rule + "]. Got error " +
        expression.failed.get.toString())
    // stick the result in the cache
    rulesCache.put(rule, expression)
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
  def calculatePseudoProbability(countMap: Map[String, Int]): (Double, Double) = {
    // prob label [sum of (weight * count)] / [sum totalRule hits]
    var weightedSum = 0.0
    var totalCount = 0.0
    var whiteListCount = 0.0
    var blackListCount = 0.0
    // for each rule
    rules.foreach {
      case (rule, weight) => {
        // did we get a hit
        if (countMap.contains(rule)) {
          val count: Int = countMap.getOrElse(rule, 0)
          if (count != 0) {
            // add to numerator
            weightedSum += (count * weight)
            // add to denominator
            totalCount += count
            // whitelist & blacklist counts
            if (weight == 1.0) {
              whiteListCount += 1.0
            } else if (weight == 0.0) {
              blackListCount += 1.0
            }
          }
        }
      }
    }
    if (whiteListCount > 0.0 || blackListCount > 0.0) {
      // if any black list or white list valued rules hit, average them,
      // else this will be 1.0 or 0.0
      (whiteListCount / (blackListCount + whiteListCount), (blackListCount + whiteListCount))
    } else {
      (weightedSum / totalCount, totalCount)
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
  def getDocumentMatchCounts(content: String): Map[String, Int] = {
    // in parallel apply the following
    rules.par.map {
      case (rule, weight) => {
        // create the regular expression
        val expression = rulesCache.get(rule)
        // create the list with matches
        val matchList = expression.flatMap(pat =>
          Try(pat.matcher(new SafeCharSequence(content, SafeCharSequence.MAX_REGEX_BACKTRACKS))))
          .map(matcher => getMatches(matcher))
        // deal with errors and create tuples to create the map from
        matchList match {
          case Success(matched) => (rule -> matched.size)
          case Failure(error) => {
            this.logger.error("Received Error: " + error.toString)
            (rule -> 0)
          }
        }
      }
      // Create list to remove parallel-ness, remove 0 count entries, then turn into a map.
    }.filter(tup => tup._2 != 0).toList.toMap
  }
}
