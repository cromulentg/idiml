package com.idibon.ml.train

import java.util

import com.idibon.ml.alloy.{ScalaJarAlloy, Alloy}
import com.idibon.ml.common.Engine
import com.idibon.ml.predict.PredictModel
import com.idibon.ml.feature.FeaturePipeline
import com.idibon.ml.predict.ensemble.EnsembleModel
import com.idibon.ml.predict.ml.IdibonLogisticRegressionModel
import com.idibon.ml.predict.rules.DocumentRules
import com.typesafe.scalalogging.StrictLogging
import org.apache.spark.ml.classification.{IdibonSparkLogisticRegressionModelWrapper, LogisticRegressionModel, LogisticRegression}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.json4s._
import org.json4s.JsonDSL._
import org.json4s.native.JsonMethods.{parse, render, compact}

import scala.collection.mutable
import scala.collection.mutable.{ListBuffer, HashMap}
import scala.util.{Try, Failure}

/** Trainer
  *
  * Performs training, given a set of documents and annotations.
  *
  * @param engine - the idiml Engine context to use for training
  */
class Trainer(engine: Engine) extends StrictLogging {

  /** Produces an RDD of LabeledPoints for each distinct label name.
    *
    * @param labeledPoints: a set of training data
    * @return a trained model based on the MLlib logistic regression model with LBFGS, trained on the provided
    *         dataset
    */
  def fitLogisticRegressionModel(labeledPoints: RDD[LabeledPoint]): LogisticRegressionModel = {
    val sqlContext = new org.apache.spark.sql.SQLContext(engine.sparkContext)
    val data = sqlContext.createDataFrame(labeledPoints)

    // set more parameters here
    val trainer = createTrainer()
    trainer.fit(data).bestModel.asInstanceOf[LogisticRegressionModel]
  }

  def createTrainer() = {
    // TODO: make these parameters more realistic
    val lr = new LogisticRegression().setMaxIter(100)
    // Print out the parameters, documentation, and any default values.
    logger.info("LogisticRegression parameters:\n" + lr.explainParams() + "\n")
    // We use a ParamGridBuilder to construct a grid of parameters to search over.
    // 6 values for lr.regParam, 6 values for elastic-net
    // 6 x 6 = 36 parameter settings for CrossValidator to choose from.
    val paramGrid = new ParamGridBuilder()
      // TODO: make these parameters more realistic
      .addGrid(lr.regParam, Array(0.01, 0.05, 0.1, 0.20, 0.35, 0.5))
      .addGrid(lr.elasticNetParam, Array(0.0, 0.2, 0.5, 0.7, 0.9, 1.0))
      .build()

    // We now treat the LR as an Estimator, wrapping it in a CrossValidator instance.
    // This will allow us to only choose parameters for the LR stage.
    // A CrossValidator requires an Estimator, a set of Estimator ParamMaps, and an Evaluator.
    // Note that the evaluator here is a BinaryClassificationEvaluator and its default metric
    // is areaUnderROC.
    val cv = new CrossValidator()
      .setEstimator(lr)
      // TODO: decide on best evaluator (this uses ROC)
      .setEvaluator(new BinaryClassificationEvaluator())
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(10) // Use 3+ in practice
    cv
  }

  private [train] def fitModels(points: Map[String, RDD[LabeledPoint]],
                                pipeline: FeaturePipeline,
                                rules: Map[String, List[(String, Float)]]):
  (Map[String, PredictModel], Map[String, String]) = {

    val labelModelMap = new mutable.HashMap[String, PredictModel]()
    val labelToUUID = new mutable.HashMap[String, String]()
    val featuresUsed = new util.HashSet[Int]()

    /* to make it easier to see everything together about the models
     * trained. Build an atomic log line. */
    val atomicLogLine = new StringBuffer()
    points.par.map {
      case (label, labeledPoints) => {
        // base LR model
        val model = fitLogisticRegressionModel(labeledPoints)
        // append info to atomic log line
        atomicLogLine.append(s"Model for $label was fit using parameters: ${model.parent.extractParamMap}\n")
        // wrap into one we want
        val wrapper = IdibonSparkLogisticRegressionModelWrapper.wrap(model)
        // create PredictModel for label:
        // LR
        val idiModel = new IdibonLogisticRegressionModel(label, wrapper, pipeline)
        // Rule
        val ruleModel = new DocumentRules(label, rules.getOrElse(label, List()))
        // Ensemble
        val ensembleModel = new EnsembleModel(label, List[PredictModel](idiModel, ruleModel))
        (label, ensembleModel, model.coefficients)
      }
      // remove parallel, and then stick it in the map
    }.toList.foreach(x => {
      labelModelMap.put(x._1, x._2)
      // TODO: UUID from task config
      labelToUUID.put(x._1, x._1)
      // add coefficients
      x._3.foreachActive((index, _) => featuresUsed.add(index))
    })
    logger.info(s"Fitted models, ${featuresUsed.size()} features used.")
    // function to pass down so that the feature transforms can prune themselves.
    // i.e. if it isn't used, remove it.
    def isNotUsed(featureIndex: Int): Boolean = {
      !featuresUsed.contains(featureIndex)
    }
    // prune unused features from global feature pipeline
    pipeline.prune(isNotUsed)
    // log training information atomically
    logger.info(atomicLogLine.toString())
    (labelModelMap.toMap, labelToUUID.toMap)
  }

  /** Trains a model and generates an Alloy from it
    *
    * Callers must provide a callback function which returns a traversable
    * list of documents; this function will be called multiple times, and
    * each invocation of the function must return an instance that will
    * traverse over the exact set of documents traversed by previous instances.
    *
    * Traversed documents should match the format generated by
    * idibin.git:/idibin/bin/open_source_integration/export_training_to_idiml.rb
    *
    *   { "content": "Who drives a chevy maliby Would you recommend it?
    *     "metadata": { "iso_639_1": "en" },
    *     "annotations: [{ "label": { "name": "Intent" }, "isPositive": true }]}
    *
    * @param pipeline - a feature pipeline to use for document processing
    * @param docs - a callback function returning a traversable sequence
    *   of JSON training documents, such as those generated by export_training_to_idiml.rb
    * @param rules a callback function returning a traversable sequence
    *   of JSON Rules, one rule per label per line, such as those generated by export_training_to_idiml.rb.
    * @param config training configuration parameters. Optional.
    * @return an Alloy with the trained model
    */
  def train(pipeline: FeaturePipeline,
            docs: () => TraversableOnce[JObject],
            rules: () => TraversableOnce[JObject],
            config: Option[JObject]): Try[Alloy] = {
    val parsedRules = rulesGenerator(rules)
    Try(RDDGenerator.getLabeledPointRDDs(this.engine, pipeline, docs))
      .flatMap{case (trainingData, featurePipeline) => {
        if (trainingData.isEmpty) {
          Failure(new IllegalArgumentException("No training data"))
        } else {
          Try(fitModels(trainingData, featurePipeline, parsedRules))
        }
      }}
      .map({ case (modelsByLabel, uuidsByLabel) => {
        new ScalaJarAlloy(modelsByLabel, uuidsByLabel)
      }})
  }

  /**
    * Creates a map of label to rules from some JSON data.
    * 
    * It expects something like {"label": label, "rule": "REGEX or TEXT", "weight": WEIGHT_VALUE}.
    *
    * @param rules a callback function returning a traversable sequence
    *   of JSON Rules, one rule per label per line, such as those generated by export_training_to_idiml.rb.
    * @return a map of label -> list of rules.
    */
  def rulesGenerator(rules: () => TraversableOnce[JObject]): Map[String, List[(String, Float)]] = {
    implicit val formats = org.json4s.DefaultFormats

    rules().map(x => {
      val label = (x \ "label").extract[String]
      val expression = (x \ "expression").extract[String]
      val weight = (x \ "weight").extract[Float]
      (label, expression, weight)
    }).foldRight(new mutable.HashMap[String, List[(String, Float)]]()){
      case ((label, expression, weight), map) => {
        val list = map.getOrElse(label, List[(String, Float)]())
        map.put(label, list :+ (expression, weight))
        map
      }
    }.toMap
  }
}
