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
    Furnace2.register[Span]("JunkFurnace", JunkFurnace)
    AlloyForge.register[Span]("JunkAlloyForge", BasicJunkAlloyForge)
    val trainerConfig = ("furnaces" -> List(
      (("name" -> "A") ~
       ("furnace" -> "JunkFurnace") ~
       ("config" -> ("delay" -> 0))),
      (("name" -> "B") ~
       ("furnace" -> "JunkFurnace") ~
       ("config" -> ("delay" -> 0)))))
    val trainer = AlloyForge[Span](new EmbeddedEngine, "JunkAlloyForge",
      "spec", Seq(new Label("00000000-0000-0000-0000-000000000000", "")),
      trainerConfig).asInstanceOf[BasicAlloyForge[Span]]
    trainer.furnaces should have length 2

    val options = TrainOptions().build(Seq())

    val alloy = Await.result(trainer.forge(options, new NoOpEvaluator()), options.maxTrainTime)
    alloy shouldBe a [BaseAlloy[_]]
    alloy.asInstanceOf[BaseAlloy[Span]].models.keys should contain theSameElementsAs Seq("A", "B")
  }

  it("should abort if model training takes too long") {
    Furnace2.register[Span]("JunkFurnace", JunkFurnace)
    AlloyForge.register[Span]("JunkAlloyForge", BasicJunkAlloyForge)
    val trainerConfig = ("furnaces" -> List(
      (("name" -> "A") ~
       ("furnace" -> "JunkFurnace") ~
       ("config" -> ("delay" -> 500)))))
    val trainer = AlloyForge[Span](new EmbeddedEngine, "JunkAlloyForge",
      "spec", Seq(new Label("00000000-0000-0000-0000-000000000000", "")),
      trainerConfig)
    val options = TrainOptions().withMaxTrainTime(0.1).build(Seq())
    intercept[TimeoutException] {
      Await.result(trainer.forge(options, new NoOpEvaluator()), Duration.Inf)
    }
  }

  it("should abort if alloy training takes too long") {
    Furnace2.register[Span]("JunkFurnace", JunkFurnace)
    AlloyForge.register[Span]("ActualJunkAlloyForge", JunkAlloyOfJunkAlloyForges)
    val trainerConfig = ("delay" -> 500)
    val trainer = AlloyForge[Span](new EmbeddedEngine, "ActualJunkAlloyForge",
      "spec", Seq(new Label("00000000-0000-0000-0000-000000000000", "")),
      trainerConfig)
    val options = TrainOptions().withMaxTrainTime(0.1).build(Seq())
    intercept[TimeoutException] {
      Await.result(trainer.forge(options, new NoOpEvaluator()), Duration.Inf)
    }
  }
}

/*
 Note: have to use Spans since we have to know the type in the basic alloy forge
 to create the right type of ensemble model, and I wasn't going to create a junk one...
 I could have mocked things but that seemed like more effort...
 */
class JunkModel extends PredictModel[Span] {
  val reifiedType = classOf[JunkModel]
  def predict(d: Document, p: PredictOptions) = Seq(new Span("", 0.0f, 0, 0, 0))
  def getFeaturesUsed = ???
  def getEvaluationMetric = ???
}

class JunkFurnace(val name: String, delay: Int) extends Furnace2[Span] {
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

class JunkAlloyForge(override val name:String, override val labels: Seq[Label], delay: Int) extends AlloyForge[Span] {
  override def doForge(options: TrainOptions, evaluator: AlloyEvaluator): Alloy[Span] = {
    implicit val context = ExecutionContext.global
    if (delay > 0) Thread.sleep(delay)
    new BaseAlloy[Span](name, labels, Map()) with HasTrainingSummary {}
  }
  override def getEvaluator(engine: Engine, taskType: String): AlloyEvaluator = ???
}

object BasicJunkAlloyForge extends ((Engine, String, Seq[Label], JObject) => AlloyForge[_]){

  override def apply(engine: Engine, name: String, labels: Seq[Label], json: JObject): BasicAlloyForge[Span] = {
    implicit val formats = org.json4s.DefaultFormats

    val config = json.extract[BasicForgeConfig]
    val furnaces = config.furnaces.map(f => {
      Furnace2[Span](engine, f.furnace, f.name, f.config)
    })
    new BasicAlloyForge[Span](name, labels, furnaces)
  }
}

object JunkAlloyForge extends ((Engine, String, Seq[Label], JObject) => AlloyForge[_]){

  override def apply(engine: Engine, name: String, labels: Seq[Label], json: JObject): BasicAlloyForge[Span] = {
    implicit val formats = org.json4s.DefaultFormats

    val config = json.extract[BasicForgeConfig]
    val furnaces = config.furnaces.map(f => {
      Furnace2[Span](engine, f.furnace, f.name, f.config)
    })
    new BasicAlloyForge[Span](name, labels, furnaces)
  }
}


object JunkAlloyOfJunkAlloyForges extends ((Engine, String, Seq[Label], JObject) => AlloyForge[_]){
  override def apply(engine: Engine, name: String, labels: Seq[Label], json: JObject): CrossValidatingAlloyForge[Span] = {
    implicit val formats = org.json4s.DefaultFormats

    val config = json.extract[JunkForgeConfig]
    val alloyForge = new JunkAlloyForge(name, labels, config.delay)
    new CrossValidatingAlloyForge[Span](engine, name, labels, alloyForge, 2, 1.0, 1L)
  }
}

case class JunkForgeConfig(delay: Int)
