package com.idibon.ml.predict.rules

import java.util.concurrent.ConcurrentHashMap
import java.util.regex.{Matcher, Pattern}

import com.idibon.ml.alloy.Alloy.{Reader, Writer}
import com.idibon.ml.alloy.Codec
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
class DocumentRules(var label: String, var rules: List[(String, Float)])
  extends RulesModel with StrictLogging {
  val invalidRules = rules.filter(x => x._2 < 0.0 || x._2 > 1.0)
  if (invalidRules.size > 0) logger.info("Filtered out rules with invalid weights: " + invalidRules.toString())
  // filter out rules that aren't valid.
  rules = rules.filter(x => x._2 >= 0.0 && x._2 <= 1.0)
  var ruleWeightMap = rules.toMap
  // Rules cache - can contain "Errors"
  val rulesCache: ConcurrentHashMap[String, Try[Pattern]] = new ConcurrentHashMap[String, Try[Pattern]]()
  // Compile the rules and populate the rule cache; this could be slow, but there's no way around this...
  populateCache()

  /**
    * Constructor for making load easy.
    */
  def this() = {
    this("", List[(String, Float)]())
  }

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
    * @param options Object of predict options.
    * @return
    */
  override def predict(features: Vector,
                       options: PredictOptions): PredictResult = {
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
  override def getFeaturesUsed(): Vector = new SparseVector(1, Array(0), Array(0))

  /** Reloads the object from the Alloy
    *
    * @param reader location within Alloy for loading any resources
    *               previous preserved by a call to
    *               { @link com.idibon.ml.feature.Archivable#save}
    * @param config archived configuration data returned by a previous
    *               call to { @link com.idibon.ml.feature.Archivable#save}
    * @return this object
    */
  override def load(reader: Reader, config: Option[JObject]): DocumentRules.this.type = {
    // it was not compiling without this implicit line...  ¯\_(ツ)_/¯
    implicit val formats = org.json4s.DefaultFormats
    this.label = (config.get \ "label").extract[String]
    val jsonObject: JValue = parse(Codec.String.read(reader.resource(RULE_RESOURCE_NAME)))
    val ruleJsonValue = jsonObject.extract[List[Map[String, Float]]]
    this.rules = ruleJsonValue.flatMap(x => x.toList)
    this.ruleWeightMap = rules.toMap
    this.rulesCache.clear()
    populateCache()
    this
  }

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
    // create output stream to write to
    val output = writer.resource(RULE_RESOURCE_NAME)
    // render the list into a json list of maps.
    val jsonString = compact(render(rules))
    logger.debug(jsonString)
    // write to the output stream via the codec.
    Codec.String.write(output, jsonString)
    Some(new JObject(List(JField("label", JString(this.label)))))
  }

  /**
    * The method used to predict from a FULL DOCUMENT!
    *
    * The model needs to handle "featurization" here.
    *
    * @param document the JObject to pull from.
    * @param options Object of predict options.
    * @return
    */
  override def predict(document: JObject,
                       options: PredictOptions): PredictResult = {
    // Takes $document out of the JObject and runs rules over them.
    val content: String = (document \ "content").asInstanceOf[JString].s
    docPredict(content, options.options.getOrElse(
      PredictOption.SignificantFeatures, false).asInstanceOf[Boolean].booleanValue())
  }

  /**
    * Predicts on a piece of content.
    * @param content
    * @param significantFeatures
    * @return
    */
  def docPredict(content: String, significantFeatures: Boolean): SingleLabelDocumentResult = {
    val dpr = new SingleLabelDocumentResultBuilder(this.getType(), this.label)
    val matchesCount: Map[String, Int] = getDocumentMatchCounts(content)
    // calculate pseudo prob.
    val (psuedoProb, totalCount, whiteOrBlackRule) = calculatePseudoProbability(matchesCount)
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
        .map(x => (x._1 -> this.ruleWeightMap.getOrElse(x._1, -1.0f))).toList
    } else {
      List()
    }
    dpr.setProbability(psuedoProb)
      .addSignificantFeatures(sigFeatures)
      .setMatchCount(totalCount)
      .setFlags(PredictResultFlag.FORCED, whiteOrBlackRule)
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
  private def savePatternToCache(rule: String, expression: Try[Pattern]): Unit = {
    if (expression.isFailure)
      this.logger.error("Failed to compile expression [${rule}]: " + expression.failed.get.toString)
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
  def getDocumentMatchCounts(content: String): Map[String, Int] = {

    // in parallel apply the following & flatten
    rules.par.flatMap {
      case (rule, weight) => {
        //get the rule or create a Failure to continue with chaining functions
        rulesCache.getOrDefault(rule, Failure(new IllegalStateException("No such rule in cache map.")))
          .flatMap(
            pat =>  // create a matcher & catch any exceptions
              Try(pat.matcher(new SafeCharSequence(content, SafeCharSequence.MAX_REGEX_BACKTRACKS)))
          ).map(getMatches(_)) //get matches into a list
          .map(rule -> _.size) //create tuples of rule -> count of matches
          .recoverWith({
          //deal with all the rules we failed to apply
          case error =>
            this.logger.error(s"Failed to apply rule:[${rule}]; " + error.toString)
            Failure(error) // since everything else returns a Try this needs to be a Failure
        }).toOption //changes this into a Some
      }
      // remove 0 counts, remove parallel-ness, make it a map
    }.filter(tup => tup._2 != 0).toList.toMap
  }

  /**
    * Override equals so that we can make unit tests simpler.
    * @param that
    * @return
    */
  override def equals(that: scala.Any): Boolean = {
    that match {
      case that: DocumentRules => {
        this.label == that.label && this.rules.equals(that.rules)
      }
      case _ => false
    }
  }

  private val RULE_RESOURCE_NAME: String = "rules.json"
}
