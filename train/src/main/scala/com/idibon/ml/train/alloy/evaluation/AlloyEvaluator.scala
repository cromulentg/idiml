package com.idibon.ml.train.alloy.evaluation

import java.util

import com.idibon.ml.alloy.Alloy
import com.idibon.ml.common.Engine
import com.idibon.ml.feature.Buildable
import com.idibon.ml.feature.tokenizer.Token
import com.idibon.ml.predict.crf.{BIOLabel, BIOType}
import com.idibon.ml.predict.ml.TrainingSummary
import com.idibon.ml.predict.ml.metrics._
import com.idibon.ml.predict.{Classification, PredictOptions, PredictOptionsBuilder, Span}
import com.idibon.ml.train.alloy.TrainingDataSet
import com.idibon.ml.train.datagenerator.crf.BIOTagger
import com.idibon.ml.train.datagenerator.json.{Document}
import org.apache.spark.mllib.evaluation.{BinaryClassificationMetrics, MulticlassMetrics, MultilabelMetrics}
import org.apache.spark.sql.functions._
import org.json4s.JsonAST.JObject

import scala.collection.JavaConversions._

/**
  * @author "Stefan Krawczyk <stefan@idibon.com>" on 3/3/16.
  */
/**
  * Alloy evaluator that handles producing training summaries for alloys.
  *
  */
trait AlloyEvaluator extends MetricHelper {

  val engine: Engine // we have a dependence on this if we use Spark's metrics computation.
  /**
    * Creates a tuple of (prediction(s), label(s)).
    * This is because this is the raw data point that spark's internal metrics
    * classes use, which we use to compute statistics from.
    *
    * @param labelToDouble
    * @param goldSet
    * @param classifications
    * @param thresholds
    */
  def createEvaluationDataPoint(labelToDouble: Map[String, Double],
                                goldSet: Map[String, Seq[EvaluationAnnotation]],
                                classifications: util.List[_],
                                thresholds: Map[String, Float]): Option[EvaluationDataPoint]

  /**
    * Creates a training summary from a sequence of data points.
    *
    * @param engine
    * @param dataPoints
    * @param labelToDouble
    * @param summaryName
    * @param portion
    * @return
    */
  def createTrainingSummary(engine: Engine,
                            dataPoints: Seq[EvaluationDataPoint],
                            labelToDouble: Map[String, Double],
                            summaryName: String,
                            portion: Double = 1.0): Seq[TrainingSummary]


  /**
    * Helper method to create metrics for each label using the raw probabilities.
    *
    * @param labelToDouble
    * @param dataPoints
    * @param metricClass
    * @return
    */
  def createPerLabelMetricsFromProbabilities(engine: Engine,
                                             labelToDouble: Map[String, Double],
                                             dataPoints: Seq[EvaluationDataPoint],
                                             metricClass: MetricClass.Value):
  Seq[Metric with Buildable[_, _]] = {
    val doubleToLabel = labelToDouble.map(x => (x._2, x._1))
    val labelProbs: Seq[Metric with Buildable[_, _]] = collatePerLabelProbabilities(
      dataPoints.flatMap(e => e.rawProbabilities), doubleToLabel, metricClass)
    val labelThresholds: Seq[Metric with Buildable[_, _]] = getSuggestedLabelThreshold(
      engine, dataPoints, doubleToLabel, metricClass)
    labelProbs ++ labelThresholds
  }

  /**
    * Helper method to create LabelProbabilities metrics.
    *
    * @param dataPoints
    * @param doubleToLabel
    * @param metricClass
    * @return
    */
  def collatePerLabelProbabilities(dataPoints: Seq[(Double, Float)],
                                   doubleToLabel: Map[Double, String],
                                   metricClass: MetricClass.Value): Seq[LabelFloatListMetric] = {
    val doubleLabelToProb = dataPoints
      .groupBy({ case (label, prob) => label })
    // create sequence of metrics, one per label
    doubleLabelToProb.map({ case (doubleLabel, probs) =>
      val points = probs.map(x => x._2).sortBy(x => x)
      new LabelFloatListMetric(
        MetricTypes.LabelProbabilities, metricClass, doubleToLabel(doubleLabel), points)
    }).toSeq
  }

  /**
    * Returns a sequence of label float metrics containing the suggested threshold for that label.
    *
    * For a label creates "binary label" data from the gold data and uses the probabilities passed in
    * to compute the threshold that achieves the optimum F1.
    *
    * @param engine
    * @param points
    * @param doubleToLabel
    * @param metricClass
    * @return
    */
  def getSuggestedLabelThreshold(engine: Engine,
                                 points: Seq[EvaluationDataPoint],
                                 doubleToLabel: Map[Double, String],
                                 metricClass: MetricClass.Value): Seq[LabelFloatMetric] = {
    val sqlContext = new org.apache.spark.sql.SQLContext(engine.sparkContext)
    val byLabel = points.flatMap(e => {
      e.rawProbabilities.map({ case (dblLabel, prob) =>
        val binaryLabel = if (e.gold.contains(dblLabel)) 1.0 else 0.0
        (dblLabel, prob, binaryLabel)
      })
    }).groupBy({ case (dblLabel, prob, binaryLabel) => dblLabel })
    byLabel.map({ case (label, grouped) =>
      val bestThreshold = computeBestF1Threshold(engine, sqlContext,
        grouped.map({ case (dblLabel, prob, binaryLabel) => (prob.toDouble, binaryLabel) }))
      new LabelFloatMetric(
        MetricTypes.LabelBestF1Threshold, metricClass, doubleToLabel(label), bestThreshold)
    }).toSeq
  }

  /**
    * Helper method to extract the float threshold value.
    *
    * Constructs the RDD & delegates to Spark's BinaryClassificationMetrics before doing
    * the query to find the right threshold.
    *
    * @param engine
    * @param sqlContext
    * @param labelValues
    * @return
    */
  def computeBestF1Threshold(engine: Engine,
                             sqlContext: org.apache.spark.sql.SQLContext,
                             labelValues: Seq[(Double, Double)]): Float = {
    val predictionRDDs = engine.sparkContext.parallelize(labelValues)
    // use 100 since that's what spark uses internally
    val binaryMetrics = new BinaryClassificationMetrics(predictionRDDs, 100)
    val fMeasure = sqlContext.createDataFrame(binaryMetrics.fMeasureByThreshold())
    // _1 is threshold, _2 is metric
    val maxFMeasure = fMeasure.select(max("_2")).head().getDouble(0)
    val bestThreshold = fMeasure.where(fMeasure.col("_2") === maxFMeasure)
      .select("_1").head().getDouble(0)
    bestThreshold.toFloat
  }

  /**
    * Evalautes an alloy using the train & test sets in the passed in data set.
    *
    * @param name    the name to give the training summaries
    * @param alloy   the alloy to test
    * @param dataSet the data set to get data from
    * @return a sequence of training summaries
    */
  def evaluate(name: String, alloy: Alloy[_], dataSet: TrainingDataSet): Seq[TrainingSummary] = {
    val uuidStrToDouble = dataSet.info.labelToDouble.map(x => (x._1.uuid.toString, x._2))
    // get thresholds
    val thresholds = alloy.getSuggestedThresholds().map({
      case (label, thresh) => (label.uuid.toString, thresh.toFloat)
    }).toMap
    val foldName = s"$name-${dataSet.info.fold}-${dataSet.info.portion}"
    //test set
    val testDataPoints = generateEvaluationPoints(dataSet.test, alloy, uuidStrToDouble, thresholds)
    val testDPs = if (testDataPoints.nonEmpty) {
      this.createTrainingSummary(engine, testDataPoints, uuidStrToDouble, s"$foldName-TEST", dataSet.info.portion)
    } else {
      Seq()
    }
    //train set
    val trainDataPoints = generateEvaluationPoints(dataSet.train, alloy, uuidStrToDouble, thresholds)
    val trainDPs = this.createTrainingSummary(engine, trainDataPoints, uuidStrToDouble, s"$foldName-TRAIN", dataSet.info.portion)
    val result = Seq(testDPs, trainDPs)
    result.flatten
  }

  /**
    * Helper method to create evaluation data points.
    *
    * @param docs            the documents to use for getting gold data and evaluating on.
    * @param alloy           the alloy to test
    * @param uuidStrToDouble map of UUID to double label for computing metrics.
    * @param thresholds      map of thresholds of UUID to threshold
    * @return a sequence of EvaluationDataPoints
    */
  def generateEvaluationPoints(docs: () => TraversableOnce[JObject],
                               alloy: Alloy[_],
                               uuidStrToDouble: Map[String, Double],
                               thresholds: Map[String, Float]) = {
    docs().map(doc => {
      val goldSet = getGoldSet(doc)
      /* FIXME: Span Rules don't return tokens or tags, thus if picked as the winning
         span, they will lower token and token tags metrics (i.e. recall). */
      val predicted = alloy.predict(doc, AlloyEvaluator.EVALUATE_PREDICT_DEFAULT)
      // create eval data point
      this.createEvaluationDataPoint(uuidStrToDouble, goldSet, predicted, thresholds)
    }).flatten.toSeq
  }

  /**
    * Gets the gold value from the evaluation set.
    *
    * @param jsValue
    * @return
    */
  def getGoldSet(jsValue: JObject): Map[String, Seq[EvaluationAnnotation]] = {
    implicit val formats = org.json4s.DefaultFormats
    val document = jsValue.extract[Document]
    document.annotations
      .filter(_.isPositive)
      .map(annot => {
        (annot.label.name, Seq(
          new EvaluationAnnotation(
            annot.label.name, annot.isPositive, annot.offset, annot.length, None, None)))
      }).toMap
  }
}

object AlloyEvaluator {
  /** constant for use in a notes metric type */
  val GRANULARITY: String = "Granularity"
  val EVALUATE_PREDICT_DEFAULT: PredictOptions = new PredictOptionsBuilder()
    .showTokens().showTokenTags().build
}

/**
  * Enum of granularities to associate in the notes metric type with.
  */
object Granularity extends Enumeration {
  type Granularity = Value
  val Document,
  Span,
  Token,
  TokenTag
  = Value
}

/**
  * Trait for creating classification based data points.
  *
  * Takes care of making sure we're dealing with classifications.
  */
trait ClassificationEvaluator {

  /**
    * Creates a tuple of (prediction(s), label(s), Seq(raw probabilities)).
    * This is because this is the raw data point that spark's internal metrics
    * classes use, which we use to compute statistics from.
    *
    * @param labelToDouble
    * @param goldSet
    * @param classifications
    * @param thresholds
    */
  def createEvaluationDataPoint(labelToDouble: Map[String, Double],
                                goldSet: Map[String, Seq[EvaluationAnnotation]],
                                classifications: util.List[_],
                                thresholds: Map[String, Float]): Option[EvaluationDataPoint] = {
    doCreateEvaluationDataPoint(
      labelToDouble, goldSet, classifications.map(_.asInstanceOf[Classification]), thresholds)
  }

  /**
    * Creates a tuple of (prediction(s), label(s), Seq(raw probabilities)).
    * This is because this is the raw data point that spark's internal metrics
    * classes use, which we use to compute statistics from.
    *
    * @param labelToDouble
    * @param goldSet
    * @param classifications
    * @param thresholds
    * @return
    */
  def doCreateEvaluationDataPoint(labelToDouble: Map[String, Double],
                                  goldSet: Map[String, Seq[EvaluationAnnotation]],
                                  classifications: Seq[Classification],
                                  thresholds: Map[String, Float]): Option[EvaluationDataPoint]
}

/**
  * Trait for creating span based data points.
  *
  * Takes care of making sure we're dealing with spans.
  */
trait SpanEvaluator {

  /**
    * Creates a tuple of (prediction(s), label(s), Seq(raw probabilities)).
    * This is because this is the raw data point that spark's internal metrics
    * classes use, which we use to compute statistics from.
    *
    * @param labelToDouble
    * @param goldSet
    * @param classifications
    * @param thresholds
    */
  def createEvaluationDataPoint(labelToDouble: Map[String, Double],
                                goldSet: Map[String, Seq[EvaluationAnnotation]],
                                classifications: util.List[_],
                                thresholds: Map[String, Float]): Option[EvaluationDataPoint] = {
    doCreateEvaluationDataPoint(
      labelToDouble, goldSet, classifications.map(_.asInstanceOf[Span]), thresholds)
  }

  /**
    * Creates a tuple of (prediction(s), label(s), Seq(raw probabilities)).
    * This is because this is the raw data point that spark's internal metrics
    * classes use, which we use to compute statistics from.
    *
    * @param labelToDouble
    * @param goldSet
    * @param span
    * @param thresholds
    * @return
    */
  def doCreateEvaluationDataPoint(labelToDouble: Map[String, Double],
                                  goldSet: Map[String, Seq[EvaluationAnnotation]],
                                  span: Seq[Span],
                                  thresholds: Map[String, Float]): Option[EvaluationDataPoint]
}

/**
  * This handles the mutually exclusive label case.
  *
  * @param defaultThreshold
  */
case class MultiClassMetricsEvaluator(override val engine: Engine, defaultThreshold: Float)
  extends AlloyEvaluator with ClassificationEvaluator {
  /**
    * Creates a tuple of (prediction(s), label(s), Seq(raw probabilities)).
    * This is because this is the raw data point that spark's internal metrics
    * classes use, which we use to compute statistics from.
    *
    * @param labelToDouble
    * @param goldSet
    * @param classifications
    * @param thresholds
    */
  def doCreateEvaluationDataPoint(labelToDouble: Map[String, Double],
                                           goldSet: Map[String, Seq[EvaluationAnnotation]],
                                           classifications: Seq[Classification],
                                           thresholds: Map[String, Float]): Option[EvaluationDataPoint] = {
    // degenerate case -- we should always have a positive polarity item for the multi-class case.
    if (goldSet.isEmpty) return None
    val goldLabel = goldSet.map({ case (gl, _) => labelToDouble(gl) }).head
    val maxLabel: Classification = getMaxLabel(classifications, thresholds)
    val rawProbabilities = classifications
      .map(c => (labelToDouble(c.label), c.probability))
    Some(new ClassificationEvaluationDataPoint(
      Array[Double](labelToDouble(maxLabel.label)), Array[Double](goldLabel), rawProbabilities))
  }

  /**
    * Handles returning the maximal label. In the mutually exclusive case we NEED
    * to return a label for every prediction. Since by definition one of the labels
    * must be chosen.
    *
    * @param classifications
    * @param thresholds
    * @return
    */
  def getMaxLabel(classifications: util.List[_],
                  thresholds: Map[String, Float]): Classification = {
    /*
      Take arg-max of ones over threshold, if none, then take plain arg-max.
      We do this weird thing because we could be dealing with a k-binary
      classifiers, and since we're simulating a multiclass classifier we NEED
      to output a prediction.
    */
    val labelsOverThreshold = classifications
      .map(c => c.asInstanceOf[Classification])
      .filter(c => {
        val threshold = thresholds.getOrDefault(c.label, defaultThreshold)
        c.probability >= threshold
      })
    val maxLabel = labelsOverThreshold.nonEmpty match {
      case true => labelsOverThreshold.maxBy(c => c.probability)
      case false => classifications.maxBy(c => c.asInstanceOf[Classification].probability)
    }
    maxLabel.asInstanceOf[Classification]
  }

  /**
    * Creates a training summary from a sequence of data points.
    *
    * @param engine
    * @param dataPoints
    * @param labelToDouble
    * @param summaryName
    * @param portion
    * @return
    */
  override def createTrainingSummary(engine: Engine,
                                     dataPoints: Seq[EvaluationDataPoint],
                                     labelToDouble: Map[String, Double],
                                     summaryName: String,
                                     portion: Double = 1.0): Seq[TrainingSummary] = {
    val labeledPoints = dataPoints.map(e => (e.predicted.head, e.gold.head))
    val predictionRDDs = engine.sparkContext.parallelize(labeledPoints)
    val multiClass = new MulticlassMetrics(predictionRDDs)
    val metrics = createMultiClassMetrics(multiClass, labelToDouble.map(x => (x._2, x._1)))
    Seq(
      new TrainingSummary(
        summaryName,
        metrics ++
          Seq[Metric with Buildable[_, _]](
            new FloatMetric(MetricTypes.Portion, MetricClass.Multiclass, portion.toFloat),
            new PropertyMetric(MetricTypes.Notes, MetricClass.Multiclass,
              Seq((AlloyEvaluator.GRANULARITY, Granularity.Document.toString)))) ++
          createPerLabelMetricsFromProbabilities(engine, labelToDouble, dataPoints, MetricClass.Multiclass)))
  }
}

/**
  * Handles the multi-label case.
  *
  * @param defaultThreshold
  */
case class MultiLabelMetricsEvaluator(override val engine: Engine, defaultThreshold: Float)
  extends AlloyEvaluator with ClassificationEvaluator {
  /**
    * Creates a tuple of (prediction(s), label(s)).
    * This is because this is the raw data point that spark's internal metrics
    * classes use, which we use to compute statistics from.
    *
    * @param labelToDouble
    * @param goldSet
    * @param classifications
    * @param thresholds
    */
  def doCreateEvaluationDataPoint(labelToDouble: Map[String, Double],
                                         goldSet: Map[String, Seq[EvaluationAnnotation]],
                                         classifications: Seq[Classification],
                                         thresholds: Map[String, Float]): Option[EvaluationDataPoint] = {
    val goldLabels = goldSet.map({ case (gl, _) => labelToDouble(gl) }).toArray
    val predictedLabels = // filter to only those over threshold -- this could potentially be empty.
      classifications.filter(c => {
        c.probability >= thresholds.getOrDefault(c.label, defaultThreshold)
      }).map(c => labelToDouble(c.label)).toArray
    val rawProbabilities = classifications
      .map(c => (labelToDouble(c.label), c.probability)).toSeq
    Some(new ClassificationEvaluationDataPoint(predictedLabels, goldLabels, rawProbabilities))
  }

  /**
    * Creates a training summary from a sequence of data points.
    *
    * @param engine
    * @param dataPoints
    * @param labelToDouble
    * @param summaryName
    * @param portion
    * @return
    */
  override def createTrainingSummary(engine: Engine,
                                     dataPoints: Seq[EvaluationDataPoint],
                                     labelToDouble: Map[String, Double],
                                     summaryName: String,
                                     portion: Double = 1.0): Seq[TrainingSummary] = {
    val labeledPoints = dataPoints.map(e => (e.predicted, e.gold))
    val predictionRDDs = engine.sparkContext.parallelize(labeledPoints)
    val multiLabel = new MultilabelMetrics(predictionRDDs)
    val metrics = createMultilabelMetrics(multiLabel, labelToDouble.map(x => (x._2, x._1)))
    Seq(
      new TrainingSummary(
        summaryName,
        metrics ++
          Seq[Metric with Buildable[_, _]](
            new FloatMetric(MetricTypes.Portion, MetricClass.Multilabel, portion.toFloat),
            new PropertyMetric(MetricTypes.Notes, MetricClass.Multilabel,
              Seq((AlloyEvaluator.GRANULARITY, Granularity.Document.toString)))) ++
          createPerLabelMetricsFromProbabilities(engine, labelToDouble, dataPoints, MetricClass.Multilabel)))
  }
}

/**
  * Evaluates BIO tagged spans.
  *
  * @param engine
  * @param bioGenerator the token & tag generator that the alloy/furnace used to generate the sequence
  *                     of possible tokens & tags from annotations.
  */
case class BIOSpanMetricsEvaluator(override val engine: Engine, bioGenerator: BIOTagger)
  extends AlloyEvaluator with SpanEvaluator {

  /**
    * Returns the gold set for span evaluation.
    *
    * This uses the passed in BIOTagger to map annotations to possible tokens
    * & tags to use for evaluation.
    *
    * @param jsValue
    * @return
    */
  override def getGoldSet(jsValue: JObject): Map[String, Seq[EvaluationAnnotation]] = {
    implicit val formats = org.json4s.DefaultFormats
    val document = jsValue.extract[Document]
    val anns = document.annotations
      .filter(ann => ann.isPositive && ann.length.get > 0 && ann.label.name.nonEmpty)
      .sortBy(_.offset.get)
    // create tokens
    val tokens = bioGenerator.sequenceGenerator(jsValue)
    // create token tag pairing from annotations
    val tokenTags = bioGenerator.getTokenTags(tokens, document)
    val newAnnotations = tokenTags
      // just collect tags with an annotation -- OUTSIDE tags don't have annotations
      .collect({ case (tok, tag: BIOLabel, Some(ann)) => (tok, tag, ann) })
      // group by annotation so we can stick everything in the right annotation
      .groupBy({ case (tok, tag, ann) => ann })
      // create new annotations based off possible tokens and tags returned
      .map({ case (ann, grouped) =>
      val sortedGroup = grouped.toSeq.sortBy({ case (tok, _, _) => tok.offset })
      val (tokenz, bioLabelz, _) = sortedGroup.unzip3
      val start = tokenz.head.offset
      val length = tokenz.map(tok => tok.length).sum
      val tagz = bioLabelz.map(b => b.bio)
      new EvaluationAnnotation(
        ann.label.name, ann.isPositive, Some(start), Some(length), Some(tokenz), Some(tagz))
    })
    // create map of label to sequence of annotations
    newAnnotations
      .groupBy(a => a.label.name)
      .map({ case (label, anns) => (label, anns.toSeq) })
  }

  /**
    * Creates a span evaluation data point -- that can house spans, tokens & tag data points.
    *
    * @param labelToDouble
    * @param goldSet
    * @param spans
    * @param thresholds
    */
  def doCreateEvaluationDataPoint(labelToDouble: Map[String, Double],
                                  goldSet: Map[String, Seq[EvaluationAnnotation]],
                                  spans: Seq[Span],
                                  thresholds: Map[String, Float]):
  Option[EvaluationDataPoint] = {
    val exactMatchCounts = exactMatchEvalDataPoint(labelToDouble, goldSet, spans)
    /*
      Predicted & gold here are for exact matches.
      They are arrays of doubles - where the first half are the classes of the labels predicted,
      while the second half is a 0 or 1 indicating whether that was correct or not.
      So from predicted we can compute tp & fp. While from gold we can compute tp & fn.
     */
    val predicted = exactMatchCounts.pmc.labels ++ exactMatchCounts.pmc.count.map(_.toDouble)
    val gold = exactMatchCounts.gmc.classes ++ exactMatchCounts.gmc.count.map(_.toDouble)
    // token & token tag datapoints are just counts of
    // tag/doubleLabel -> (true pos, false pos, false neg) counts.
    val tokenDP: Seq[TokenDataPoint] = tokenEvalDataPoint(labelToDouble, goldSet, spans)
    val tokenTagDP: Seq[TokenTagDataPoint] = tokenTagEvalDataPoint(labelToDouble, goldSet, spans)
    val predictedProbability = exactMatchCounts.pp.map(p => (p.label, p.probability))
    Some(
      new SpanEvaluationDataPoint(
        predicted.toArray, gold.toArray, predictedProbability, tokenDP, tokenTagDP))
  }

  /**
    * Creates data points for exact matching of spans.
    *
    * They are arrays of doubles - where the first half are the classes of the labels predicted,
    * while the second half is a 0 or 1 indicating whether that was correct or not.
    * So from predicted we can compute tp & fp. While from gold we can compute tp & fn.
    *
    * @param labelToDouble
    * @param goldSet
    * @param spans
    * @return Triple of tuples:
    *         ((double label, count of matches),
    *         (predicted classes, correctness),
    *         (gold classes, correctly guessed))
    */
  def exactMatchEvalDataPoint(labelToDouble: Map[String, Double],
                              goldSet: Map[String, Seq[EvaluationAnnotation]],
                              spans: Seq[Span]) = {
    val numberGuessesCorrect: Iterable[(Double, Int, Float)] = spans.map(s => {
      val matches = goldSet.getOrElse(s.label, Seq()).filter(a => {
        s.offset == a.offset.get && s.length == a.length.get
      })
      require(matches.size <= 1, "Should not match multiple annotations.")
      (labelToDouble(s.label), matches.size, s.probability)
    })
    val byLabel = spans.groupBy(s => s.label)
    // note: don't flat map straight from the map because the compiler creates a map, rather than a sequence...
    val numberGoldCorrect: Seq[(Double, Int)] = goldSet.toSeq.flatMap({ case (label, annotations) =>
      annotations.map(a => {
        val matches = byLabel.getOrElse(label, Seq()).filter(s => {
          s.offset == a.offset.get && s.length == a.length.get
        })
        require(matches.size <= 1, "Should not match multiple span predictions.")
        (labelToDouble(label), matches.size)
      })
    })
    val (doubleLabel, matches, probabilities) = numberGuessesCorrect.unzip3
    val predictedProbabilities = doubleLabel.zip(probabilities)
      .map({case (label, prob) => PredictedProbability(label, prob)}).toSeq
    val numGoldCorrect = numberGoldCorrect.unzip
    ExactMatchCounts(
      PredictedMatchCounts(doubleLabel, matches),
      GoldMatchCounts(numGoldCorrect._1, numGoldCorrect._2),
      predictedProbabilities)
  }

  /**
    * Creates data points for token evaluation.
    *
    * The algorithm is basically comparing the predicted sequence with the gold sequence.
    * First we group by label, and then check the predicted and gold tokens based on token offsets.
    *
    * Then depending on how things match up, we increment:
    * - true positives if we match properly
    * - false positives if a prediction misses
    * - false negatives if we miss predicting a gold
    *
    * @param labelToDouble
    * @param goldSet
    * @param spans
    * @return a seq of (double label -> (tp, fp, tn) counts)
    */
  def tokenEvalDataPoint(labelToDouble: Map[String, Double],
                         goldSet: Map[String, Seq[EvaluationAnnotation]],
                         spans: Seq[Span]) = {
    /*
    1. predicted: label -> sequence of tokens
    2. gold: label -> sequence of tokens
    3. then for each label:
        - compare heads of lists:
         -- match == tp
         -- predicted miss == fp
         -- gold miss = fn
     */
    val predictedTokensByLabel = spans
      .groupBy(s => s.label)
      .map({ case (label, span) =>
        (label, span.flatMap(s => s.tokens).sortBy(t => t.offset))
      })
    val goldTokensByLabel = goldSet
      .map({ case (label, anns) =>
        (label, anns.flatMap(a => a.tokens).flatten.sortBy(t => t.offset))
      })
    labelToDouble.map({ case (label, doubleValue) =>
      // have to handle missing labels not being present
      val predicted = predictedTokensByLabel.getOrElse(label, Seq())
      val gold = goldTokensByLabel.getOrElse(label, Seq())
      val (tp, fp, fn) = tokenDPHelper(predicted, gold)
      new TokenDataPoint(doubleValue, tp, fp, fn)
    }).toSeq
  }

  /**
    * This is a helper function to compare a sequence of tokens.
    *
    * We assume that we have already prefiltered all tokens to a single label/value that
    * they represent. Otherwise The algorithm is basically comparing the predicted sequence
    * with the gold sequence.
    *
    * This entails checking the offsets, namely:
    * 1) that we're advancing over the right list based on the offset
    * 2) and that we're incrementing the right case based on things matching or not.
    *
    * @param predictedSeq predicted sequence of tokens to compare
    * @param goldSeq gold sequence of tokens to compare
    * @return (true positive count, false positive count, false negative count)
    */
  def tokenDPHelper(predictedSeq: Seq[Token], goldSeq: Seq[Token]) = {
    var predicted = predictedSeq
    var gold = goldSeq
    val sc = StatsCounter(0, 0, 0)

    while (predicted.nonEmpty && gold.nonEmpty) {
      val pHead = predicted.head
      val gHead = gold.head
      if (pHead.offset < gHead.offset) {
        // prediction is before
        sc.fp += 1
        predicted = predicted.tail
      } else if (gHead.offset < pHead.offset) {
        // gold is before
        sc.fn += 1
        gold = gold.tail
      } else {
        // we're at the same spot
        require(pHead.length == gHead.length, "Token lengths should be the same at the same offset.")
        require(pHead.content.equals(gHead.content), "Token content should be the same at the same offset.")
        sc.tp += 1
        gold = gold.tail
        predicted = predicted.tail
      }
    }
    sc.fp += predicted.size
    sc.fn += gold.size
    // true positives -- the number of tokens guessed correctly
    // false positives -- the number of tokens we guessed that aren't in the gold
    // false negatives -- the number of tokens we missed guessing that are in gold
    (sc.tp, sc.fp, sc.fn)
  }

  /**
    * Creates data points for token tag evaluation.
    *
    * The algorithm is basically comparing the predicted sequence with the gold sequence.
    * First we group by the tag, and then check the predicted and gold tokens & tags
    * based on token offsets.
    *
    * Then depending on how things match up, we increment:
    * - true positives if we match properly
    * - false positives if a prediction misses
    * - false negatives if we miss predicting a gold
    *
    * @param labelToDouble
    * @param goldSet
    * @param spans
    * @return a map of (tag name -> (tp, fp, tn) counts)
    */
  def tokenTagEvalDataPoint(labelToDouble: Map[String, Double],
                            goldSet: Map[String, Seq[EvaluationAnnotation]],
                            spans: Seq[Span]) = {
    /*
    Create:
    1. predicted: tag -> sequence of (token, tags)
    2. gold: tag -> sequence of (token, tags)
    3. then for each tag:
        - compare heads of lists:
         -- match == tp
         -- predicted miss == fp
         -- gold miss = fn
     */
    val predictedTags: Map[BIOType.Value, Seq[(Token, BIOType.Value)]] = spans
      .flatMap(span => span.getTokenNTags())
      .sortBy(t => t._1.offset)
      .groupBy({case (token, tag) => tag})

    val goldTags: Map[BIOType.Value, Seq[(Token, BIOType.Value)]] = goldSet.toSeq
      .flatMap({ case (_, anns) => anns.flatMap(a => a.getTokensNTags())})
      .sortBy(x => x._1.offset)
      .groupBy({case (token, tag) => tag})

    BIOType.values.map(bio => {
      val predicted = predictedTags.getOrElse(bio, Seq())
      val gold = goldTags.getOrElse(bio, Seq())
      val (tp, fp, fn) = tokenDPHelper(
        predicted.map({case (token, _) => token}),
        gold.map({case (token, _) => token}))
      new TokenTagDataPoint(bio.toString, tp, fp, fn)
      // remove all zeros
    }).toSeq.filter(t => t.fn > 0 || t.fp > 0 || t.tp > 0)
  }

  /**
    * Creates training summaries from a sequence of data points.
    *
    * It extracts the three sets of data points - spans, tokens & token tags.
    *
    * @param engine
    * @param dataPoints
    * @param labelToDouble
    * @param summaryName
    * @param portion
    * @return
    */
  override def createTrainingSummary(engine: Engine,
                                     dataPoints: Seq[EvaluationDataPoint],
                                     labelToDouble: Map[String, Double],
                                     summaryName: String, portion: Double): Seq[TrainingSummary] = {
    val dbleToLabel = labelToDouble.map(x => (x._2, x._1))
    val exactSummary = new TrainingSummary(summaryName,
      new FloatMetric(MetricTypes.Portion, MetricClass.Multiclass, portion.toFloat) +:
      new PropertyMetric(
        MetricTypes.Notes,
        MetricClass.Multiclass,
        Seq((AlloyEvaluator.GRANULARITY, Granularity.Span.toString))) +:
        exactMatchEval(dataPoints.map(dp => (dp.predicted, dp.gold)), dbleToLabel))
    val tokenMetrics = tokenMatchEval(dataPoints
      .map(dp => dp.asInstanceOf[SpanEvaluationDataPoint])
      .flatMap(dp => dp.tokenDP), labelToDouble)
    val tokenSummary = new TrainingSummary(
      summaryName,
      new FloatMetric(MetricTypes.Portion, MetricClass.Multiclass, portion.toFloat) +:
      new PropertyMetric(
        MetricTypes.Notes,
        MetricClass.Multiclass,
        Seq((AlloyEvaluator.GRANULARITY, Granularity.Token.toString))) +: tokenMetrics)
    val tokenTagMetrics = tokenTagMatchEval(dataPoints
      .map(dp => dp.asInstanceOf[SpanEvaluationDataPoint])
      .flatMap(dp => dp.tokenTagDP))
    val tokenTagSummary = new TrainingSummary(
      summaryName,
      new FloatMetric(MetricTypes.Portion, MetricClass.Multiclass, portion.toFloat) +:
      new PropertyMetric(
        MetricTypes.Notes,
        MetricClass.Multiclass,
        Seq((AlloyEvaluator.GRANULARITY, Granularity.TokenTag.toString))) +: tokenTagMetrics)
    Seq(exactSummary, tokenSummary, tokenTagSummary)
  }

  /**
    * From token tag data points, creates metrics.
    *
    * @param tokenTagDPs
    * @return sequence of metrics
    */
  def tokenTagMatchEval(tokenTagDPs: Seq[TokenTagDataPoint]): Seq[Metric with Buildable[_, _]] = {
    val byTag = tokenTagDPs.groupBy(ttdp => ttdp.tagName)
    val totalsByTag = sumGrouped[String](byTag)
    val metricsByTag = createPRF1Metrics(computePRF1Metrics(totalsByTag))
    val microMetrics: Seq[Metric with Buildable[_, _]] = computeMicroMetrics(totalsByTag)
    val macroMetrics: Seq[Metric with Buildable[_, _]] = computeMacroMetrics(metricsByTag)
    val tagMetrics: Seq[Metric with Buildable[_, _]] = metricsByTag
      .flatMap { case (label, (p, r, f1)) => Seq(p, r, f1) }.toSeq
    tagMetrics ++ microMetrics ++ macroMetrics
  }

  /**
    * Sums a group of StatsCounts metrics across it's values
    *
    * @param grouped
    * @tparam T
    * @return
    */
  def sumGrouped[T](grouped: Map[T, Seq[StatsCounts]]) = {
    grouped.map({ case (key, sequence) =>
      val (trueP, falseP, falseN) = sequence.foldRight((0, 0, 0))({ case (seq, (ttp, tfp, tfn)) =>
        (seq.tp + ttp, seq.fp + tfp, seq.fn + tfn)
      })
      (key, (trueP, falseP, falseN))
    })
  }

  /**
    * Computes P, R & F1 metrics from a map of key -> (tp, fp, fn) counts.
    *
    * @param totals
    * @tparam T
    * @return
    */
  def computePRF1Metrics[T](totals: Map[T, (Int, Int, Int)]) = {
    totals.filter({ case (k, (tp, fp, fn)) => (tp + fp + fn) > 0 }).map({ case (key, (tp, fp, fn)) =>
      val precision = if (tp + fp == 0) {
        0f
      } else {
        tp.toFloat / (tp.toFloat + fp.toFloat)
      }
      val recall = if (tp + fn == 0) {
        0f
      } else {
        tp.toFloat / (tp.toFloat + fn.toFloat)
      }
      val f1 = computeF1(precision, recall)
      (key, (precision, recall, f1))
    })
  }

  /**
    * Creates P, R & F1 label metrics objects from a map of label -> (p, r, f1) values.
    *
    * @param metrics
    * @param mClass
    * @return
    */
  def createPRF1Metrics(metrics: Map[String, (Float, Float, Float)],
                        mClass: MetricClass.Value = MetricClass.Multiclass) = {
    metrics
      .map({ case (label, (precision, recall, f1Val)) =>
        val p = new LabelFloatMetric(MetricTypes.LabelPrecision, mClass, label, precision)
        val r = new LabelFloatMetric(MetricTypes.LabelRecall, mClass, label, recall)
        val f1m = new LabelFloatMetric(MetricTypes.LabelF1, mClass, label, f1Val)
        (label, (p, r, f1m))
      })
  }

  /**
    * Computes & creates micro { P, R & F1 } metrics from a map with values with (tp, fp, fn) counts.
    *
    * @param totals
    * @param mClass
    * @return
    */
  def computeMicroMetrics(totals: Map[_, (Int, Int, Int)],
                          mClass: MetricClass.Value = MetricClass.Multiclass) = {
    // unzip into lists
    val (tP, fP, fN) = totals.values.unzip3
    // sum those lists
    val (tPsum, fPSum, fNSum) = (tP.sum, fP.sum, fN.sum)
    // compute
    val microP = tPsum.toFloat / (tPsum.toFloat + fPSum.toFloat)
    val microR = tPsum.toFloat / (tPsum.toFloat + fNSum.toFloat)
    val microF1 = computeF1(microP, microR)
    Seq(
      new FloatMetric(MetricTypes.MicroPrecision, mClass, microP),
      new FloatMetric(MetricTypes.MicroRecall, mClass, microR),
      new FloatMetric(MetricTypes.MicroF1, mClass, microF1)
    )
  }

  /**
    * Computes and creates macro {P, R & F1} metrics from a map with values of label float metrics.
    *
    * The values should be tuples of (precision, recall & f1) -- these values are averaged to
    * give the macro versions.
    *
    * @param metrics
    * @param mClass
    * @return
    */
  def computeMacroMetrics(metrics: Map[_, (LabelFloatMetric, LabelFloatMetric, LabelFloatMetric)],
                          mClass: MetricClass.Value = MetricClass.Multiclass) = {
    // unzip to just get 3 lists that we then individually sum over
    val (macroPRaw, macroRRaw, macroF1Raw) = metrics.values.unzip3
    val macroP = macroPRaw.map(_.float).sum / metrics.size.toFloat
    val macroR = macroRRaw.map(_.float).sum / metrics.size.toFloat
    val macroF1 = macroF1Raw.map(_.float).sum / metrics.size.toFloat
    Seq(
      new FloatMetric(MetricTypes.MacroPrecision, mClass, macroP),
      new FloatMetric(MetricTypes.MacroRecall, mClass, macroR),
      new FloatMetric(MetricTypes.MacroF1, mClass, macroF1)
    )
  }

  /**
    * Computes metrics for token evaluation from a sequence of token data points.
    *
    * @param tokenDPs
    * @param labelToDouble
    * @return
    */
  def tokenMatchEval(tokenDPs: Seq[TokenDataPoint],
                     labelToDouble: Map[String, Double]): Seq[Metric with Buildable[_, _]] = {
    val dblToLabel = labelToDouble.map(x => (x._2, x._1))
    val byLabel = tokenDPs.groupBy(tkdp => tkdp.doubleLabel)
    val totalsByLabel = sumGrouped[Double](byLabel)
    val metricsByLabel = createPRF1Metrics(
      computePRF1Metrics(totalsByLabel).map({
        case (doubleLabel, numbers) => (dblToLabel(doubleLabel), numbers)
      }))
    val microMetrics: Seq[Metric with Buildable[_, _]] = computeMicroMetrics(totalsByLabel)
    val macroMetrics: Seq[Metric with Buildable[_, _]] = computeMacroMetrics(metricsByLabel)
    val labelMetrics: Seq[Metric with Buildable[_, _]] = metricsByLabel
      .flatMap { case (label, (p, r, f1)) => Seq(p, r, f1) }.toSeq
    labelMetrics ++ microMetrics ++ macroMetrics
  }

  /**
    * Computes metrics for the exact span match case.
    *
    * @param guesses
    */
  def exactMatchEval(guesses: Seq[(Array[Double], Array[Double])],
                     dbleToLabel: Map[Double, String]) = {
    // Note: can't compute accuracy since we don't have a count of true negatives
    /*
      Predicted & gold here are for exact matches.
      They are arrays of doubles - where the first half are the classes of the labels predicted,
      while the second half is a 0 or 1 indicating whether that was correct or not.
      So from predicted we can compute tp & fp. While from gold we can compute tp & fn.
     */
    val (predictedVals, goldVals) = guesses.map(dp => {
      val predicted = dp._1.slice(0, dp._1.size / 2)
      val predictedCorrect = dp._1.slice(dp._1.size / 2, dp._1.size)
      val gold = dp._2.slice(0, dp._2.size / 2)
      val goldCorrect = dp._2.slice(dp._2.size / 2, dp._2.size)
      (predicted.zip(predictedCorrect), gold.zip(goldCorrect))
    }).unzip
    val predictedByLabel = predictedVals
      .flatten
      .groupBy({ case (label, wasCorrect) => label })
      .map({ case (label, grouped) =>
        val correct = grouped.map(_._2.toInt).sum
        val size = grouped.size
        (label, (correct, size))
      })
    val goldByLabel = goldVals
      .flatten
      .groupBy({ case (label, wasCorrect) => label })
      .map({ case (label, grouped) =>
        val correct = grouped.map(_._2.toInt).sum
        val size = grouped.size
        (label, (correct, size))
      })
    val microResults: Seq[Metric with Buildable[_, _]] = microPrecisionRecallF1(predictedByLabel, goldByLabel)
    val labelResults: Seq[Metric with Buildable[_, _]] = byLabelPrecisionRecallF1(predictedByLabel, goldByLabel, dbleToLabel)
    microResults ++ labelResults
  }

  /**
    * Computes label P & R & F1 given:
    * - counts by label that:
    * -- represent how many predictions were correct
    * -- represent how many golds were corrrect
    * calculates the label P & R & F1.
    *
    * @param predictedByLabel
    * @param goldByLabel
    * @param dbleToLabel
    * @param mClass
    * @return sequence of LabelFloatMetrics
    */
  def byLabelPrecisionRecallF1(predictedByLabel: Map[Double, (Int, Int)],
                               goldByLabel: Map[Double, (Int, Int)],
                               dbleToLabel: Map[Double, String],
                               mClass: MetricClass.Value = MetricClass.Multiclass): Seq[Metric with Buildable[_, _]] = {
    predictedByLabel.map({ case (label, (correct, size)) =>
      val (_, goldSize) = goldByLabel(label)
      val precision = correct.toFloat / size.toFloat
      val recall = correct.toFloat / goldSize.toFloat
      (label, Seq(
        new LabelFloatMetric(MetricTypes.LabelPrecision, mClass, dbleToLabel(label), precision),
        new LabelFloatMetric(MetricTypes.LabelRecall, mClass, dbleToLabel(label), recall),
        new LabelFloatMetric(MetricTypes.LabelF1, mClass, dbleToLabel(label), computeF1(precision, recall))
      ))
    }).flatMap(_._2).toSeq
  }

  /**
    * Computes micro P & R & F1 given:
    * - counts by label that:
    * -- represent how many predictions were correct
    * -- represent how many golds were corrrect
    * calculates the micro P & R & F1 over everything.
    *
    * @param predictedByLabel
    * @param goldByLabel
    * @param mClass
    * @return
    */
  def microPrecisionRecallF1(predictedByLabel: Map[Double, (Int, Int)],
                             goldByLabel: Map[Double, (Int, Int)],
                             mClass: MetricClass.Value = MetricClass.Multiclass) = {
    val precisionCounts = predictedByLabel.foldRight((0, 0))({ case ((_, (correct, size)), (totalCorrect, totalSize)) =>
      (correct + totalCorrect, size + totalSize)
    })
    val precision = precisionCounts._1.toFloat / precisionCounts._2.toFloat
    val recallCounts = goldByLabel.foldRight((0, 0))({ case ((_, (correct, size)), (totalCorrect, totalSize)) =>
      (correct + totalCorrect, size + totalSize)
    })
    val recall = recallCounts._1.toFloat / recallCounts._2.toFloat
    Seq(
      new FloatMetric(MetricTypes.MicroPrecision, mClass, precision),
      new FloatMetric(MetricTypes.MicroRecall, mClass, recall),
      new FloatMetric(MetricTypes.MicroF1, mClass, computeF1(precision, recall))
    )
  }

  /**
    * Computes F1 given precision and recall values.
    *
    * @param precision
    * @param recall
    * @return
    */
  def computeF1(precision: Float, recall: Float): Float = {
    if (precision + recall != 0f) {
      2.0f * (precision * recall) / (precision + recall)
    } else {
      0f
    }
  }
}

/**
  * Noop Evaluator.
  */
case class NoOpEvaluator() extends AlloyEvaluator {
  override val engine: Engine = null

  override def createEvaluationDataPoint(labelToDouble: Map[String, Double],
                                         goldSet: Map[String, Seq[EvaluationAnnotation]],
                                         classifications: util.List[_],
                                         thresholds: Map[String, Float]): Option[EvaluationDataPoint] = {
    Some(new ClassificationEvaluationDataPoint(Array(), Array(), Seq()))
  }

  override def createTrainingSummary(engine: Engine,
                                     dataPoints: Seq[EvaluationDataPoint],
                                     labelToDouble: Map[String, Double],
                                     summaryName: String,
                                     portion: Double): Seq[TrainingSummary] = Seq()
}

/**
  * Trait to help store data from an evaluation.
  *
  */
trait EvaluationDataPoint {
  def predicted: Array[Double]

  def gold: Array[Double]

  def rawProbabilities: Seq[(Double, Float)]
}

/**
  * Class that handles classification related
  *
  * @param predicted
  * @param gold
  * @param rawProbabilities
  */
case class ClassificationEvaluationDataPoint(predicted: Array[Double],
                                             gold: Array[Double],
                                             rawProbabilities: Seq[(Double, Float)])
  extends EvaluationDataPoint

/**
  * Houses counts/stats for a single data point for span evaluation.
  *
  * @param predicted
  * @param gold
  * @param rawProbabilities
  * @param tokenDP
  * @param tokenTagDP
  */
case class SpanEvaluationDataPoint(predicted: Array[Double],
                                   gold: Array[Double],
                                   rawProbabilities: Seq[(Double, Float)],
                                   tokenDP: Seq[TokenDataPoint],
                                   tokenTagDP: Seq[TokenTagDataPoint])
  extends EvaluationDataPoint

/**
  * Has:
  * True positive
  * False positive
  * False Negative
  * Counts
  */
trait StatsCounts {
  def tp: Int

  def fp: Int

  def fn: Int
}

/*
 Helper classes for making objects that get passed around easier to understand.
 */

case class StatsCounter(var tp: Int, var fp: Int, var fn: Int) extends StatsCounts

case class TokenDataPoint(doubleLabel: Double, tp: Int, fp: Int, fn: Int) extends StatsCounts

case class TokenTagDataPoint(tagName: String, tp: Int, fp: Int, fn: Int) extends StatsCounts

case class ExactMatchCounts(pmc: PredictedMatchCounts,
                            gmc: GoldMatchCounts,
                            pp: Seq[PredictedProbability])

case class PredictedMatchCounts(labels: Iterable[Double], count: Iterable[Int])
case class GoldMatchCounts(classes: Iterable[Double], count: Iterable[Int])
case class PredictedProbability(label: Double, probability: Float)
