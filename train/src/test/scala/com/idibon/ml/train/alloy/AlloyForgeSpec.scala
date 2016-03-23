package com.idibon.ml.train.alloy

import com.idibon.ml.train.alloy.evaluation.{NoOpEvaluator, AlloyEvaluator}

import scala.concurrent._
import scala.concurrent.duration._

import com.idibon.ml.predict._
import com.idibon.ml.feature._
import com.idibon.ml.alloy.{HasTrainingSummary, Alloy, BaseAlloy}
import com.idibon.ml.common.{Engine, EmbeddedEngine}
import com.idibon.ml.train.furnace.Furnace2
import com.idibon.ml.train.TrainOptions

import org.json4s.JObject
import org.json4s.JsonDSL._

import org.scalatest.{Matchers, FunSpec, BeforeAndAfter}

import scala.util.Try

class AlloyForgeSpec extends FunSpec with Matchers with BeforeAndAfter {

  before {
    Furnace2.resetRegistry()
    AlloyForge.resetRegistry()
  }

  it("should train multiple furnaces") {
    Furnace2.register[JunkResult]("JunkFurnace", JunkFurnace)
    AlloyForge.register[JunkResult]("JunkAlloyForge", BasicJunkAlloyForge)
    val trainerConfig = ("furnaces" -> List(
      (("name" -> "A") ~
       ("furnace" -> "JunkFurnace") ~
       ("config" -> ("delay" -> 0))),
      (("name" -> "B") ~
       ("furnace" -> "JunkFurnace") ~
       ("config" -> ("delay" -> 0)))))
    val trainer = AlloyForge[JunkResult](new EmbeddedEngine, "JunkAlloyForge",
      "spec", Seq(new Label("00000000-0000-0000-0000-000000000000", "")),
      trainerConfig).asInstanceOf[BasicAlloyForge[Span]]
    trainer.furnaces should have length 2

    val options = TrainOptions().build(Seq())

    val alloy = Await.result(trainer.forge(options, new NoOpEvaluator()), options.maxTrainTime)
    alloy shouldBe a [BaseAlloy[_]]
    alloy.asInstanceOf[BaseAlloy[JunkResult]].models.keys should contain theSameElementsAs Seq("A", "B")
  }

  it("should abort if model training takes too long") {
    Furnace2.register[JunkResult]("JunkFurnace", JunkFurnace)
    AlloyForge.register[JunkResult]("JunkAlloyForge", BasicJunkAlloyForge)
    val trainerConfig = ("furnaces" -> List(
      (("name" -> "A") ~
       ("furnace" -> "JunkFurnace") ~
       ("config" -> ("delay" -> 500)))))
    val trainer = AlloyForge[JunkResult](new EmbeddedEngine, "JunkAlloyForge",
      "spec", Seq(new Label("00000000-0000-0000-0000-000000000000", "")),
      trainerConfig)
    val options = TrainOptions().withMaxTrainTime(0.1).build(Seq())
    intercept[TimeoutException] {
      Await.result(trainer.forge(options, new NoOpEvaluator()), Duration.Inf)
    }
  }

  it("should abort if alloy training takes too long") {
    Furnace2.register[JunkResult]("JunkFurnace", JunkFurnace)
    AlloyForge.register[JunkResult]("ActualJunkAlloyForge", JunkAlloyOfJunkAlloyForges)
    val trainerConfig = ("delay" -> 500)
    val trainer = AlloyForge[JunkResult](new EmbeddedEngine, "ActualJunkAlloyForge",
      "spec", Seq(new Label("00000000-0000-0000-0000-000000000000", "")),
      trainerConfig)
    val options = TrainOptions().withMaxTrainTime(0.1).build(Seq())
    intercept[TimeoutException] {
      Await.result(trainer.forge(options, new NoOpEvaluator()), Duration.Inf)
    }
  }
}

class JunkResult extends PredictResult
    with Buildable[JunkResult, JunkResult]
    with Builder[JunkResult] {
  def label = ""
  def probability = 0.0f
  def matchCount = 0
  def flags = 0

  def save(os: FeatureOutputStream) {}
  def build(is: FeatureInputStream) = this
}

class JunkModel extends PredictModel[JunkResult] {
  val reifiedType = classOf[JunkModel]
  def predict(d: Document, p: PredictOptions) = Seq(new JunkResult)
  def getFeaturesUsed = ???
  def getEvaluationMetric = ???
}

class JunkFurnace(val name: String, delay: Int) extends Furnace2[JunkResult] {
  protected def doHeat(options: TrainOptions) = {
    if (delay > 0) Thread.sleep(delay)
    new JunkModel
  }
}

object JunkFurnace extends ((Engine, String, JObject) => Furnace2[_]) {
  def apply(e: Engine, n: String, c: JObject) = {
    implicit val formats = org.json4s.DefaultFormats
    val delay = (c \ "delay").extract[Int]
    new JunkFurnace(n, delay)
  }
}

class JunkAlloyForge(override val name:String, override val labels: Seq[Label], delay: Int) extends AlloyForge[JunkResult] {
  override def doForge(options: TrainOptions, evaluator: AlloyEvaluator): Alloy[JunkResult] = {
    implicit val context = ExecutionContext.global
    if (delay > 0) Thread.sleep(delay)
    new BaseAlloy[JunkResult](name, labels, Map()) with HasTrainingSummary {}
  }
  override def getEvaluator(engine: Engine, taskType: String): AlloyEvaluator = ???
}

object BasicJunkAlloyForge extends ((Engine, String, Seq[Label], JObject) => AlloyForge[_]){

  override def apply(engine: Engine, name: String, labels: Seq[Label], json: JObject): BasicAlloyForge[JunkResult] = {
    implicit val formats = org.json4s.DefaultFormats

    val config = json.extract[BasicForgeConfig]
    val furnaces = config.furnaces.map(f => {
      Furnace2[JunkResult](engine, f.furnace, f.name, f.config)
    })
    new BasicAlloyForge[JunkResult](name, labels, furnaces)
  }
}

object JunkAlloyForge extends ((Engine, String, Seq[Label], JObject) => AlloyForge[_]){

  override def apply(engine: Engine, name: String, labels: Seq[Label], json: JObject): BasicAlloyForge[JunkResult] = {
    implicit val formats = org.json4s.DefaultFormats

    val config = json.extract[BasicForgeConfig]
    val furnaces = config.furnaces.map(f => {
      Furnace2[JunkResult](engine, f.furnace, f.name, f.config)
    })
    new BasicAlloyForge[JunkResult](name, labels, furnaces)
  }
}


object JunkAlloyOfJunkAlloyForges extends ((Engine, String, Seq[Label], JObject) => AlloyForge[_]){
  override def apply(engine: Engine, name: String, labels: Seq[Label], json: JObject): CrossValidatingAlloyForge[JunkResult] = {
    implicit val formats = org.json4s.DefaultFormats

    val config = json.extract[JunkForgeConfig]
    val alloyForge = new JunkAlloyForge(name, labels, config.delay)
    new CrossValidatingAlloyForge[JunkResult](engine, name, labels, alloyForge, 2, 1.0, 1L)
  }
}

case class JunkForgeConfig(delay: Int)
