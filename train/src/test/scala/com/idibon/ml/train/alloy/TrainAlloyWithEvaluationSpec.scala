package com.idibon.ml.train.alloy

import java.util

import com.idibon.ml.alloy.Alloy
import com.idibon.ml.common
import com.idibon.ml.common.{EmbeddedEngine}
import com.idibon.ml.predict.ml.TrainingSummary
import com.idibon.ml.predict.{PredictOptions, Label, Classification}
import com.idibon.ml.train.alloy.evaluation._
import org.json4s
import org.json4s.JsonAST.JObject
import org.json4s._
import org.json4s.native.JsonMethods._
import org.scalatest._
import scala.collection.JavaConversions._
import scala.collection.JavaConverters._

/**
  * Tests the TrainAlloyWithEvaluation class
  */
class TrainAlloyWithEvaluationSpec extends FunSpec
  with Matchers with BeforeAndAfter with ParallelTestExecution with BeforeAndAfterAll {

  val engine = new EmbeddedEngine
  implicit val formats = org.json4s.DefaultFormats

  describe("create gold set tests") {
    val trainer = new TrainAlloyWithEvaluation(
      "a", engine, null, new TrainingDataSet(new DataSetInfo(0,0.0,Map()),null, null), null)
    it("handles degenerate all negatives without breaking") {
      val allNegs =
        parse("""
          {"content": "", "annotations": [{"label": {"name": "a"}, "isPositive": false},
           {"label": {"name": "b"}, "isPositive": false}]}
        """)
      val actual = trainer.getGoldSet(allNegs.extract[JObject])
      actual shouldBe Map[String, EvaluationAnnotation]()
    }
    it("handles mutli label data data") {
      val allNegs =
        parse("""
            {"content": "", "annotations": [
            {"label": {"name": "a"}, "isPositive": true},
            {"label": {"name": "b"}, "isPositive": false},
            {"label": {"name": "c"}, "isPositive": true},
            {"label": {"name": "d"}, "isPositive": false},
          ]}""")
      val actual = trainer.getGoldSet(allNegs.extract[JObject])
      actual shouldBe Map[String, Seq[EvaluationAnnotation]](
        "a" -> Seq(EvaluationAnnotation(LabelName("a"), true, None, None)),
        "c" -> Seq(EvaluationAnnotation(LabelName("c"), true, None, None)))
    }
    it("handles mutually exclusive label data data") {
      val allNegs =
        parse("""
          {"content": "", "annotations": [
            {"label": {"name": "a"}, "isPositive": false},
            {"label": {"name": "b"}, "isPositive": false},
            {"label": {"name": "c"}, "isPositive": false},
            {"label": {"name": "d"}, "isPositive": true},
          ]}""")
      val actual = trainer.getGoldSet(allNegs.extract[JObject])
      actual shouldBe Map[String, Seq[EvaluationAnnotation]](
        "d" -> Seq(EvaluationAnnotation(LabelName("d"), true, None, None)))
    }
  }
  describe("evaluate tests") {

    it("handles empty gold set"){
      val allNegs =
        parse("""
          {"content": "", "annotations": [
            {"label": {"name": "a"}, "isPositive": false},
            {"label": {"name": "b"}, "isPositive": false},
            {"label": {"name": "c"}, "isPositive": false},
            {"label": {"name": "d"}, "isPositive": false},
          ]}""").extract[JObject]
      val trainer = new TrainAlloyWithEvaluation(
        "a", engine, null, new TrainingDataSet(
          new DataSetInfo(0, 0.0, Map()),
          null, () => Seq(allNegs)), new DummyAlloyEvaluator())
      val dummy = new DummyAlloy()
      val actual = trainer.evaluate(dummy, Map("a" -> 0.3f))
      actual shouldBe Seq()
    }
    it("works as intended") {
      val onePos =
        parse("""
          {"content": "", "annotations": [
            {"label": {"name": "a"}, "isPositive": false},
            {"label": {"name": "b"}, "isPositive": false},
            {"label": {"name": "c"}, "isPositive": false},
            {"label": {"name": "d"}, "isPositive": true},
          ]}""").extract[JObject]
      val trainer = new TrainAlloyWithEvaluation(
        "a", engine, null, new TrainingDataSet(
          new DataSetInfo(0, 0.0, Map()),
          null, () => Seq(onePos)), new DummyAlloyEvaluator())
      val dummy = new DummyAlloy()
      val actual = trainer.evaluate(dummy, Map("a" -> 0.3f))
      actual.size shouldBe 1
      actual(0).predicted shouldBe Array(1.0)
      actual(0).gold shouldBe Array(1.0)
    }
  }

  describe("apply everything gels together") {
    it("runs through everything using the dummy objects") {
      val onePos =
        parse("""
          {"content": "", "annotations": [
            {"label": {"name": "a"}, "isPositive": false},
            {"label": {"name": "b"}, "isPositive": false},
            {"label": {"name": "c"}, "isPositive": false},
            {"label": {"name": "d"}, "isPositive": true},
          ]}""").extract[JObject]
      val trainer = new TrainAlloyWithEvaluation(
        "a",
        engine,
        new DummyTrainer(),
        new TrainingDataSet(
          new DataSetInfo(0, 0.0, Map()),
          null, () => Seq(onePos)),
        new DummyAlloyEvaluator())
      val Seq(actual) = trainer(null, None)
      actual.identifier shouldBe "test"
      actual.metrics shouldBe Seq()
    }
  }
}

class DummyAlloy extends Alloy[Classification] {
  override def getLabels: util.List[Label] = ???

  override def getSuggestedThresholds: util.Map[Label, java.lang.Float] = Map(
    new Label("00000000-0000-0000-0000-000000000000", "A") -> new java.lang.Float(0.4f)).asJava

  override def predict(document: JObject, options: PredictOptions): util.List[Classification] = {
    List(new Classification("a", 0.5f, 1, 0, Seq()))
  }

  override def save(writer: Alloy.Writer): Unit = ???

  override def translateUUID(uuid: String): Label = ???
}
class DummyAlloyEvaluator extends AlloyEvaluator {
  override val engine: common.Engine = null
  override def createEvaluationDataPoint(labelToDouble: Map[String, Double],
                                         goldSet: Map[String, Seq[EvaluationAnnotation]],
                                         classifications: util.List[_],
                                         thresholds: Map[String, Float]): Option[EvaluationDataPoint] = {
    if (goldSet.isEmpty) None
    else Some(new ClassificationEvaluationDataPoint(Array(1.0), Array(1.0), Seq()))
  }

  override def createTrainingSummary(engine: common.Engine,
                                     dataPoints: Seq[EvaluationDataPoint],
                                     labelToDouble: Map[String, Double],
                                     summaryName: String,
                                     portion: Double): Seq[TrainingSummary] = {
    Seq(new TrainingSummary("test", Seq()))
  }
}
class DummyTrainer extends AlloyTrainer {

  override def trainAlloy(name: String,
                          docs: () => TraversableOnce[json4s.JObject],
                          labelsAndRules: json4s.JObject,
                          config: Option[json4s.JObject]): Alloy[Classification] = new DummyAlloy()
}
