package com.idibon.ml.predict.rules

import java.util.regex.Matcher

import com.idibon.ml.alloy.Alloy.{Writer, Reader}
import com.idibon.ml.alloy.Codec
import com.idibon.ml.common.{Archivable, Engine, ArchiveLoader}
import com.idibon.ml.predict.util.SafeCharSequence
import com.idibon.ml.predict._
import com.typesafe.scalalogging.StrictLogging
import org.apache.spark.mllib.linalg.{SparseVector, Vector}
import org.json4s._
import org.json4s.native.JsonMethods._
import org.json4s.JsonDSL.WithDouble._

import scala.util.{Failure, Success, Try}

/**
  * Class that handles span rules.
  *
  * @author "Stefan Krawczyk <stefan@idibon.com>" on 3/24/16.
  * @param labelUUID the UUID label this set of rules represents.
  * @param labelHuman the human label for use in named group identificaiton if used.
  * @param rules the rules with weights to use.
  */
case class SpanRules(labelUUID: String, labelHuman: String, rules: List[(String, Float)])
  extends PredictModel[Span] with StrictLogging  with RuleStorage
    with Archivable[SpanRules, SpanRulesLoader]{

  def getLogger = logger
  val reifiedType: Class[_ <: PredictModel[Span]] = classOf[SpanRules]

  /**
    * The model will use a subset of features passed in. This method
    * should return the ones used.
    *
    * @return Vector (likely SparseVector) where indices correspond to features
    *         that were used.
    */
  override def getFeaturesUsed(): Vector = new SparseVector(1, Array(0), Array(0))

  /**
    * After training, this method returns the value of the metric commonly used to evaluate
    * model performance
    *
    * @return Double (e.g. AreaUnderROC)
    */
  override def getEvaluationMetric(): Double = ???

  /**
    * The method used to predict from a vector of features.
    *
    * @param document Document that contains the original JSON.
    * @param options  Object of predict options.
    * @return
    */
  override def predict(document: Document, options: PredictOptions): Seq[Span] = {
    // Takes $document out of the JObject and runs rules over them.
    val content: String = (document.json \ "content").asInstanceOf[JString].s
    // NOTE: we don't observe any predict options at the moment.
    spanPredict(content)
  }

  /**
    * Creates the spans for a given piece of content.
    *
    * @param content the string content to look at.
    * @return possibly empty sequence of spans
    */
  def spanPredict(content: String): Seq[Span] = {
    // since we either got the whole thing, or a named group, we don't have any overlaps
    // to worry about
    val matches = getRuleMatches(content)
    // create spans
    matches.flatMap({case (rule, matchIndexes) =>
      val weight = ruleWeightMap(rule)
      val flagValue = if (weight == 0f || weight == 1.0f) {
        PredictResultFlag.mask(PredictResultFlag.FORCED, PredictResultFlag.RULE)
      } else {
        PredictResultFlag.mask(PredictResultFlag.RULE)
      }
      matchIndexes.map({case (start, end) =>
        new Span(labelUUID, weight, flagValue, start, end - start)
      })
    }).toSeq
  }

  /**
    * For each rule, get the start and end of any matches.
    *
    * This cycles through all the rules and tries to match on the content.
    * It then gets all the matches, returning for each rule, the start and
    * end position of each match of that rule.
    *
    * @param content the content to match on
    * @return
    */
  def getRuleMatches(content: String): Map[String, Seq[(Int, Int)]] = {
    // in parallel apply the following
    rulesCache.par.map({ case (rule, pat) => {
      pat.flatMap(p => {
        /* create a matcher for the content given the compiled pattern,
         * and locate all matches; catch any errors thrown (e.g., due to
         * excessive backtracking */
        Try{
          val matcher = p.matcher(
            new SafeCharSequence(content, SafeCharSequence.MAX_REGEX_BACKTRACKS))
          getMatches(matcher, labelHuman)
        }
      })
        .filter(matchList => matchList.nonEmpty)
        .map(matchList => rule -> matchList)
      /* only return successful matches that are found
       * FIXME: this should return errors, so that they can be reported
       * to the caller
       */
    }}).filter(result => result.isSuccess)
      .map(_.get).toMap.seq
  }

  /**
    * Finds all the matches and returns their starting and ending indexes.
    *
    * @param matcher the already matched pattern object.
    * @param namedGroup the string name of the group to look for on each match.
    * @return a list of integer tuples - (startIndex, endIndex)
    */
  def getMatches(matcher: Matcher, namedGroup: String): List[(Int, Int)] = {
    var matches: List[(Int, Int)] = List[(Int, Int)]()
    while (matcher.find()) {
      //TODO: do some JVM hackery when compiling rules to also save the named group index
      // This should hopefully lead to a perf improvement...
      val groupIndex = getMatcherGroupIndex(matcher, namedGroup)
      val start = matcher.start(groupIndex)
      val end = matcher.end(groupIndex)
      // append to the end of the list
      matches = matches :+(start, end)
    }
    matches
  }

  /**
    * Returns the matcher group index to use when using named groups.
    *
    * Note: 0 == the capturing group that represents the full regular
    * expression (all matched text). This is the default value if
    * no capturing group named "namedGroup" exists in the expression.
    *
    * @param matcher the already matched pattern object.
    * @param namedGroup the string name of the group to look.
    * @return 0 if not found, else the index of the named group.
    */
  def getMatcherGroupIndex(matcher: Matcher, namedGroup: String): Int = {
    /* The regex state machine throws an IllegalArgumentException if there
     * is no named capturing group for the requested name; in that case,
     * suppress the exception and fall-through to the default return */
    val matchedText = Try(matcher.group(namedGroup))
    matchedText match {
      case Failure(f) => 0
      case Success(mt) => if (mt.nonEmpty) {
        /* In a *bizarre* API oversight, Java has no efficient mechanism for
        * lookup up the group index that corresponds to a named capturing
        * group, so iterate over all capturing groups and see which one
        * exactly matches the captured text from the named group.
        #
        * FIXME: this algorithm works for reasonable regular expressions;
        * however, if the expression does something absurd, like define
        * the capturing group as a back reference, e.g.:
        * /(walla)\s+(?P<label>\1)/, such that the earlier group index
        * has *exactly the same matching text* as the named capturing
        * group, the returned index will be for the earlier text.
        * I don't seee a good way around this limitation given Java's API,
        * and this seems like a bad idea */
        (1 to matcher.groupCount).foreach(group => {
          if (matcher.group(group) != null &&
            matcher.group(group).equals(mt)) return group
        })
      }
      0
    }
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
    // render the list into a json list of maps.
    val jsonString = compact(render(rules))
    // create output stream to write to
    val output = writer.resource(SpanRules.RULE_RESOURCE_NAME)
    try {
      // write to the output stream via the codec.
      Codec.String.write(output, jsonString)
      Some(new JObject(List(
        JField("labelUUID", JString(this.labelUUID)),
        JField("labelHuman", JString(this.labelHuman))
      )))
    } finally {
      // close the stream
      output.close()
    }
  }
}

class SpanRulesLoader extends ArchiveLoader[SpanRules] {
  /** Reloads the object from the Alloy
    *
    * @param engine implementation of the Engine trait
    * @param reader location within Alloy for loading any resources
    *               previous preserved by a call to
    *               { @link com.idibon.ml.common.Archivable#save}
    * @param config archived configuration data returned by a previous
    *               call to { @link com.idibon.ml.common.Archivable#save}
    * @return this object
    */
  override def load(engine: Engine, reader: Option[Reader], config: Option[JObject]): SpanRules = {
    // it was not compiling without this implicit line...  ¯\_(ツ)_/¯
    implicit val formats = org.json4s.DefaultFormats
    val labelUUID = (config.get \ "labelUUID").extract[String]
    val labelHuman = (config.get \ "labelHuman").extract[String]
    val jsonObject: JValue = parse(
      Codec.String.read(reader.get.resource(SpanRules.RULE_RESOURCE_NAME)))

    val ruleJsonValue = jsonObject.extract[List[Map[String, Float]]]
    val rules = ruleJsonValue.flatMap(x => x.toList)
    new SpanRules(labelUUID, labelHuman, rules)
  }
}


private[this] object SpanRules {
  val RULE_RESOURCE_NAME: String = "rules.json"
}
