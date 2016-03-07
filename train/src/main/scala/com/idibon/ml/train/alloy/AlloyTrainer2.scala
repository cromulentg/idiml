package com.idibon.ml.train.alloy

import scala.util.Try
import scala.concurrent.{ExecutionContext, Future, Promise}
import scala.reflect.runtime.universe.TypeTag

import com.idibon.ml.train.TrainOptions
import com.idibon.ml.train.furnace.{Furnace2, HasFurnaces}
import com.idibon.ml.predict.{Label, PredictResult}
import com.idibon.ml.feature.{Buildable, Builder}
import com.idibon.ml.alloy.{Alloy, BaseAlloy}
import com.idibon.ml.common.Engine

import org.json4s.JObject

/** Container class for generating Alloys from one or more model trainers
  *
  * @tparam T the prediction result type of this alloy
  * @param name a name for the alloy
  * @param labels the set of all labels assignable from the alloy
  * @param furnaces furnaces used to generate the models stored in the alloy
  */
sealed class AlloyTrainer2[T <: PredictResult with Buildable[T, Builder[T]]](
  val name: String,
  val labels: Seq[Label],
  override val furnaces: Seq[Furnace2[T]])
    extends HasFurnaces[T] {

  /** Initiate a batch training process across all models to produce the Alloy
    *
    * @param options configuration options
    * @return an asynchronous training Future with the Alloy
    */
  def train(options: TrainOptions): Future[Alloy[T]] = {
    implicit val context = ExecutionContext.global

    val alloy = Promise[Alloy[T]]()
    Future {
      // train all of the models, creating a name => Model map for the alloy
      val models = Try(furnaces.map(_.name).zip(trainFurnaces(options)).toMap)
      alloy.complete(models.flatMap(m => Try(new BaseAlloy(name, labels, m))))
    }
    alloy.future
  }
}

object AlloyTrainer2 {

  /** Initialize a new trainer with the provided configuration
    *
    * @tparam T expected prediction result type
    * @param engine current engine context
    * @param name name for the generated alloy
    * @param labels labels assigned by the alloy
    * @param json configuration data for the training process
    */
  def apply[T <: PredictResult with Buildable[T, Builder[T]]: TypeTag](
    engine: Engine, name: String, labels: Seq[Label], json: JObject):
      AlloyTrainer2[T] = {
    implicit val formats = org.json4s.DefaultFormats

    val config = json.extract[AlloyTrainerConfig]
    val furnaces = config.furnaces.map(f => {
      Furnace2[T](engine, f.furnace, f.name, f.config)
    })
    new AlloyTrainer2[T](name, labels, furnaces)
  }
}

// === JSON configuration schema ===
case class AlloyTrainerConfig(furnaces: List[FurnaceConfig])

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
