package com.idibon.ml.train.alloy

import com.idibon.ml.alloy.{HasTrainingSummary, BaseAlloy, Alloy}
import com.idibon.ml.common.Engine
import com.idibon.ml.feature.{SequenceGenerator, Builder, Buildable}
import com.idibon.ml.predict.{Classification, Span, Label, PredictResult}
import com.idibon.ml.train.TrainOptions
import com.idibon.ml.train.alloy.evaluation.{NoOpEvaluator, BIOSpanMetricsEvaluator, AlloyEvaluator}
import com.idibon.ml.train.furnace.{ChainNERFurnace, HasFurnaces, Furnace2}
import com.typesafe.scalalogging.StrictLogging
import org.json4s._

import scala.concurrent.{Promise, ExecutionContext, Future}
import scala.util.{Failure, Success, Try}

/** Basic class for generating Alloys from one or more furnaces.
  *
  * @tparam T the prediction result type of this alloy
  * @param name a name for the alloy
  * @param labels the set of all labels assignable from the alloy
  * @param furnaces furnaces used to generate the models stored in the alloy
  */
sealed class BasicAlloyForge[T <: PredictResult with Buildable[T, Builder[T]]](
     override val name: String,
     override val labels: Seq[Label],
     override val furnaces: Seq[Furnace2[T]])
  extends AlloyForge[T] with HasFurnaces[T] with StrictLogging {

  /** Initiate a batch training process across all models to produce the Alloy
    *
    * @param options configuration options
    * @param evaluator the evaluator used to measure this alloy.
    * @return an asynchronous training Future with the Alloy
    */
  def doForge(options: TrainOptions, evaluator: AlloyEvaluator): Alloy[T] = {
    implicit val context = ExecutionContext.global

    // train all of the models, creating a name => Model map for the alloy
    val models = furnaces.map(_.name).zip(heatFurnaces(options)).toMap
    // create wrapper alloy for evaluation
    val baseAlloy = new BaseAlloy[T](name, labels, models)
    // evaluate
    val summaries = evaluator.evaluate(name, baseAlloy, options.dataSet)
    // now make real alloy with results
    new BaseAlloy(name, labels, models) with HasTrainingSummary {
      override def getTrainingSummaries = if (summaries.nonEmpty) Some(summaries) else None
    }
  }

  /**
    * Returns the appropriate evaluator for this alloy.
    *
    * @param engine
    * @param taskType
    * @return
    */
  override def getEvaluator(engine: Engine, taskType: String): AlloyEvaluator = {
    taskType match {
      case "extraction.bio_ner" => {
        val biotagger = furnaces
          .filter(f => f.isInstanceOf[ChainNERFurnace])
          .map(f => f.asInstanceOf[ChainNERFurnace])
          .head
        new BIOSpanMetricsEvaluator(engine, biotagger)
      }
      case _ => new NoOpEvaluator()
    }
  }
}

object BasicClassificationAlloyForge extends ((Engine, String, Seq[Label], JObject) => AlloyForge[_]){

  /** Initialize a new trainer with the provided configuration
    *
    * @param engine current engine context
    * @param name name for the generated alloy
    * @param labels labels assigned by the alloy
    * @param json configuration data for the training process
    */
  override def apply(engine: Engine, name: String, labels: Seq[Label], json: JObject):
  BasicAlloyForge[Classification] = {
    implicit val formats = org.json4s.DefaultFormats

    val config = json.extract[BasicForgeConfig]
    val furnaces = config.furnaces.map(f => {
      Furnace2[Classification](engine, f.furnace, f.name, f.config)
    })
    new BasicAlloyForge[Classification](name, labels, furnaces)
  }
}

object BasicSpanAlloyForge extends ((Engine, String, Seq[Label], JObject) => AlloyForge[_]){

  /** Initialize a new trainer with the provided configuration
    *
    * @param engine current engine context
    * @param name name for the generated alloy
    * @param labels labels assigned by the alloy
    * @param json configuration data for the training process
    */
  override def apply(engine: Engine, name: String, labels: Seq[Label], json: JObject):
  BasicAlloyForge[Span] = {
    implicit val formats = org.json4s.DefaultFormats

    val config = json.extract[BasicForgeConfig]
    val furnaces = config.furnaces.map(f => {
      Furnace2[Span](engine, f.furnace, f.name, f.config)
    })
    new BasicAlloyForge[Span](name, labels, furnaces)
  }
}

// === JSON configuration schema ===
case class BasicForgeConfig(furnaces: List[FurnaceConfig])

/** Schema for each furnace used within the trainer
  *
  * All furnaces must produce the same result type (i.e., all Span, or all
  * Classification)
  *
  * @param name name of the generated alloy
  * @param furnace name for the furnace type
  * @param config configuration JSON for the furnace
  */
case class FurnaceConfig(name: String, furnace: String, config: JObject)
