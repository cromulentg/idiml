package com.idibon.ml.predict.ensemble

import com.idibon.ml.predict.ml.metrics._

import scala.collection.mutable.HashMap

import com.idibon.ml.alloy._
import com.idibon.ml.common.{Engine, ArchiveLoader, Archivable, EmbeddedEngine}
import com.idibon.ml.feature._
import com.idibon.ml.predict.ml.{TrainingSummary}
import com.idibon.ml.predict.rules.{SpanRules}
import com.idibon.ml.predict._
import org.apache.spark.mllib.linalg.{Vectors, Vector}
import org.json4s._
import org.json4s.JsonDSL._
import org.scalatest.{BeforeAndAfter, FunSpec, Matchers}

/**
  * Class to test the Span Ensemble Model
  */
class SpanEnsembleModelSpec extends FunSpec with Matchers with BeforeAndAfter {

  val pipeline: FeaturePipeline = (FeaturePipelineBuilder.named("StefansPipeline")
    += (FeaturePipelineBuilder.entry("metaExtractor", new MetadataNumberExtractor2, "$document"))
    := ("metaExtractor"))

  val text: String = "Everybody loves replacing hadoop with spark because it is much faster. a b d"
  val doc: JObject = JObject("content" -> text, "metadata" -> JObject(List(("number", JDouble(0.5)))))
  val fp = pipeline.prime(List(doc))

  describe("reduce tests") {

  }

  describe("save and load") {
    it("should save and load properly, using reified types") {
      val archive = HashMap[String, Array[Byte]]()
      val spanRules1 = new SpanRules("alabel", "halabel", List(("loves", 0.5f))) {
        def uselessAnonymousClass = "break getClass calls"
      }
      val spanRules2 = new SpanRules("blabel", "hblabel", List(("is", 0.5f)))
      val ensembleModel1 = new SpanEnsembleModel("test",
        Map("alabel" -> spanRules1, "blabel" -> spanRules2))
      val metadata = ensembleModel1.save(new MemoryAlloyWriter(archive))
      val expectedMetadata = Some(JObject(List(
        ("name", JString("test")),
        ("labels", JArray(List(JString("alabel"), JString("blabel")))),
        ("model-meta", JObject(List(
          ("alabel", JObject(List(
            ("config", JObject(List(("label", JObject(List(
              ("uuid", JString("alabel")),
              ("human", JString("halabel"))
            )))))),
            ("class", JString("com.idibon.ml.predict.rules.SpanRules"))))),
          ("blabel", JObject(List(
            ("config", JObject(List(("label", JObject(List(
              ("uuid", JString("blabel")),
              ("human", JString("hblabel"))
            )))))),
            ("class", JString("com.idibon.ml.predict.rules.SpanRules"))))
            ))))
      )))
      metadata shouldBe expectedMetadata
      val ensembleModel2 = (new SpanEnsembleModelLoader).load(
        new EmbeddedEngine, Some(new MemoryAlloyReader(archive.toMap)), metadata)
      val e1Pred = ensembleModel1.predict(Document.document(doc),
        new PredictOptionsBuilder().build())
      val e2Pred = ensembleModel2.predict(Document.document(doc),
        new PredictOptionsBuilder().build())
      e1Pred shouldBe e2Pred
    }

    it("should save and load empty models") {
      val archive = HashMap[String, Array[Byte]]()
      val spanRules1 = new SpanRules("alabel", "halabel", List())
      val spanRules2 = new SpanRules("blabel", "hblabel", List())
      val ensembleModel1 = new SpanEnsembleModel("test",
        Map("alabel" -> spanRules1, "blabel" -> spanRules2))
      val metadata = ensembleModel1.save(new MemoryAlloyWriter(archive))
      val expectedMetadata = Some(JObject(List(
        ("name", JString("test")),
        ("labels", JArray(List(JString("alabel"), JString("blabel")))),
        ("model-meta", JObject(List(
          ("alabel", JObject(List(
            ("config", JObject(List(("label", JObject(List(
              ("uuid", JString("alabel")),
              ("human", JString("halabel"))
            )))))),
            ("class", JString("com.idibon.ml.predict.rules.SpanRules"))))),
          ("blabel", JObject(List(
            ("config", JObject(List(("label", JObject(List(
              ("uuid", JString("blabel")),
              ("human", JString("hblabel"))
            )))))),
            ("class", JString("com.idibon.ml.predict.rules.SpanRules"))))
            ))))
      )))
      metadata shouldBe expectedMetadata
      val ensembleModel2 = (new SpanEnsembleModelLoader).load(
        new EmbeddedEngine, Some(new MemoryAlloyReader(archive.toMap)), metadata)
      val e1Pred = ensembleModel1.predict(Document.document(doc),
        new PredictOptionsBuilder().build())
      val e2Pred = ensembleModel2.predict(Document.document(doc),
        new PredictOptionsBuilder().build())
      e1Pred shouldBe e2Pred
    }
  }
  describe("training summary test cases") {
    it("should collate training summaries from underlying models"){
      val summary1 = Some(Seq(new TrainingSummary("testing1",
        Seq[Metric with Buildable[_, _]](
          new FloatMetric(MetricTypes.AreaUnderROC, MetricClass.Binary, 0.5f),
          new PointsMetric(MetricTypes.F1ByThreshold, MetricClass.Binary,
            Seq((0.4f, 0.5f), (0.3f, 0.2f)))))))
      val summary2 = Some(Seq(new TrainingSummary("testing2",
        Seq[Metric with Buildable[_, _]](
          new FloatMetric(MetricTypes.F1, MetricClass.Binary, 0.7f),
          new PointsMetric(MetricTypes.F1ByThreshold, MetricClass.Binary,
            Seq((0.4f, 0.5f), (0.3f, 0.2f)))))))
      val ensem1 = new SpanEnsembleModel("test", Map(
        "0" -> new FakeMCModel2(List("alabel", "blabel")),
        "1" -> new FakeMCModel2(List("alabel", "blabel")) {
          override val trainingSummary = summary1
        },
        "2" -> new FakeMCModel2(List("alabel", "blabel")) {
          override val trainingSummary = summary2
        }))
      val actual = ensem1.getTrainingSummary().get.sortBy(ts => ts.identifier)
      val expected = summary1.get ++ summary2.get
      actual shouldBe expected

    }
    it("should return none when no summaries exist") {
      val ensem1 = new SpanEnsembleModel("test", Map(
        "0" -> new FakeMCModel2(List("alabel", "blabel"))))
      ensem1.getTrainingSummary() shouldBe None
    }

    it("should return none when no underlying model summaries exist") {
      val gangSummary = new TrainingSummary("gang",
        Seq(new FloatMetric(MetricTypes.Recall, MetricClass.Multiclass, 0.5f)))
      val ensem1 = new SpanEnsembleModel("test", Map(
        "0" -> new FakeMCModel2(List("alabel", "blabel")))) {
        override val trainingSummary = Some(Seq(gangSummary))
      }
      ensem1.getTrainingSummary() shouldBe Some(Seq(gangSummary))
    }

    it("should collate training summaries with underlying models"){
      val summary1 = Some(Seq(new TrainingSummary("testing1",
        Seq[Metric with Buildable[_, _]](
          new FloatMetric(MetricTypes.AreaUnderROC, MetricClass.Binary, 0.5f),
          new PointsMetric(MetricTypes.F1ByThreshold, MetricClass.Binary,
            Seq((0.4f, 0.5f), (0.3f, 0.2f)))))))
      val summary2 = Some(Seq(new TrainingSummary("testing2",
        Seq[Metric with Buildable[_, _]](
          new FloatMetric(MetricTypes.F1, MetricClass.Binary, 0.7f),
          new PointsMetric(MetricTypes.F1ByThreshold, MetricClass.Binary,
            Seq((0.4f, 0.5f), (0.3f, 0.2f)))))))
      val gangSummary = new TrainingSummary("gang",
        Seq(new FloatMetric(MetricTypes.Recall, MetricClass.Multiclass, 0.5f)))
      val ensem1 = new SpanEnsembleModel("test", Map(
        "0" -> new FakeMCModel2(List("alabel", "blabel")),
        "1" -> new FakeMCModel2(List("alabel", "blabel")) {
          override val trainingSummary = summary1
        },
        "2" -> new FakeMCModel2(List("alabel", "blabel")) {
          override val trainingSummary = summary2
        })) {
        override val trainingSummary=Some(Seq(gangSummary))
      }
      val actual = ensem1.getTrainingSummary().get.sortBy(ts => ts.identifier)
      val expected = Seq(gangSummary) ++ summary1.get ++ summary2.get
      actual shouldBe expected
    }
  }

  describe("document prediction test cases") {
    it("should work as expected with a single fake model") {
      val ensem1 = new SpanEnsembleModel("test", Map[String, PredictModel[Span]](
        "0" -> new FakeMCModel2(List("alabel", "blabel"))))
      val doc = new JObject(List("content" -> new JString("string matching is working"),
        "metadata" -> JObject(List(("number", JDouble(0.5))))))
      val actual = ensem1.predict(Document.document(doc), PredictOptions.DEFAULT)
      actual.map(_.label) shouldBe List("alabel", "blabel")
      actual.map(_.matchCount) shouldBe List(1, 1)
      actual.map(_.probability) shouldEqual List(0.2f, 0.2f)
    }
    it("should handle merging 'predicted' and rule spans -- only rules should be returned") {
      val spanRules = new SpanRules("blabel", "hblabel", List(("/str[ij]ng/", 0.6f), ("is", 0.6f)))
      val ensem1 = new SpanEnsembleModel("test", Map[String, PredictModel[Span]](
        "0" -> new FakeMCModel2(List("alabel", "blabel")), "1" -> spanRules))
      val doc = new JObject(List("content" -> new JString("string matching is working"),
        "metadata" -> JObject(List(("number", JDouble(0.5))))))
      val actual = ensem1.predict(Document.document(doc), PredictOptions.DEFAULT)
      actual.map(_.label) shouldBe List("blabel", "blabel")
      actual.map(_.matchCount) shouldBe List(1, 1)
      actual.map(_.probability) shouldEqual List(0.6f, 0.6f)
      actual.map(_.flags) shouldEqual List(2, 2)
    }
    it("works as intended returning predicted when rules don't fire") {
      val spanRules1 = new SpanRules("blabel", "hblabel", List(("/sadfadsf/", 1.0f), ("asdasdf", 0.6f)))
      val spanRules2 = new SpanRules("alabel", "halabel", List(("/safsadf/", 0.0f), ("asdfasdf", 0.6f)))
      val ensem1 = new SpanEnsembleModel("test", Map[String, PredictModel[Span]](
        "0" -> new FakeMCModel2(List("alabel", "blabel")),
        "1" -> spanRules1, "2" -> spanRules2))
      val doc = new JObject(List("content" -> new JString("string matching is working"),
        "metadata" -> JObject(List(("number", JDouble(0.5))))))
      val actual = ensem1.predict(Document.document(doc),
        new PredictOptionsBuilder().build())
      actual.map(_.label) shouldBe List("alabel", "blabel")
      // since we whitelist/blacklist - we drop all the other model results, hence match count of 1.
      actual.map(_.matchCount) shouldBe List(1, 1)
      actual.map(_.probability) shouldEqual List(0.2f, 0.2f)
    }
    it("works when nothing matches") {
      val spanRules1 = new SpanRules("blabel", "hblabel", List(("/sadfadsf/", 1.0f), ("asdasdf", 0.6f)))
      val spanRules2 = new SpanRules("alabel", "halabel", List(("/safsadf/", 0.0f), ("asdfasdf", 0.6f)))
      val ensem1 = new SpanEnsembleModel("test", Map[String, PredictModel[Span]](
        "1" -> spanRules1, "2" -> spanRules2))
      val doc = new JObject(List("content" -> new JString("string matching is working"),
        "metadata" -> JObject(List(("number", JDouble(0.5))))))
      val actual = ensem1.predict(Document.document(doc),
        new PredictOptionsBuilder().build())
      actual.isEmpty shouldBe true
    }
  }

  describe("reduce tests") {
    val ensem1 = new SEMExtension("test", Map())
    it("works on empty sequence") {
      ensem1.reduce(Seq()) shouldBe Seq()
    }
    it("handles overlaping spans") {
      // a overlaps with all, but b should be chosen and currently due to greedy algo. c is dropped.
      val predictions = Seq(
        Span("a", 0.6f, 0, 2, 10), Span("b", 1.0f, 3, 5, 2), Span("c", 0.5f, 2, 10, 10))
      ensem1.reduce(predictions) shouldBe Seq(predictions(1))
    }
    it("removes rule spans below 0.5 in weight") {
      val predictions = Seq(
        // rule, rule + force, "predicted"
        Span("a", 0.3f, 2, 2, 2), Span("b", 0.3f, 3, 5, 2), Span("c", 0.4f, 0, 10, 10))
      ensem1.reduce(predictions) shouldBe Seq(predictions(2))
    }
  }
}

class SEMExtension(name: String,
                   models: Map[String, PredictModel[Span]])
  extends SpanEnsembleModel(name, models) {

  override def reduce(predictions: Seq[Span]): Seq[Span] = super.reduce(predictions)
}

/**
  * Fake class to make it easy to test combining results.
 *
  * @param labels
  */
case class FakeMCModel2(labels: List[String])
    extends PredictModel[Span]
    with Archivable[FakeMCModel2, FakeMCModelLoader2] {

  val reifiedType = classOf[FakeMCModel2]

  def predict(document: Document, options: PredictOptions): Seq[Span] = {
    labels.map(label => label match {
      case "alabel" => Span(label, 0.2f, 0, 0, 2)
      case _ => Span(label, 0.2f, 0, 5, 2)
    })
  }

  override def getTrainingSummary(): Option[Seq[TrainingSummary]] = {
    trainingSummary
  }

  override def getFeaturesUsed(): Vector = ???

  override def getEvaluationMetric(): Double = ???

  override def save(writer: Alloy.Writer): Option[JObject] = {
    val out = writer.resource("stuff")
    Codec.VLuint.write(out, labels.size)
    labels.foreach(label => Codec.String.write(out, label))
    out.close()
    None
  }
}

class FakeMCModelLoader2 extends ArchiveLoader[FakeMCModel2] {
  override def load(engine: Engine, reader: Option[Alloy.Reader], config: Option[JObject]): FakeMCModel2 = {
    val in = reader.get.resource("stuff")
    val size = Codec.VLuint.read(in)
    val labels = (0 until size).map(_ => {
      val label = Codec.String.read(in)
      label
    })
    in.close()
    new FakeMCModel2(labels.toList)
  }
}

/**
  * Helps create a more manageable feature pipeline.
  */
private [this] class MetadataNumberExtractor2 extends FeatureTransformer with TerminableTransformer {
  def apply(document: JObject): Vector = {
    val num: Double = (document \ "metadata" \ "number").asInstanceOf[JDouble].num
    Vectors.dense(Array(num, num, num))
  }
  var pruned = false

  def freeze(): Unit = {}

  def getFeatureByIndex(index: Int) = index match {
    case 0 => Some(StringFeature("meta-number1"))
    case 1 => Some(StringFeature("meta-number2"))
    case 2 => Some(StringFeature("meta-number3"))
    case _ => None
  }

  def numDimensions = Some(3)

  def prune(transform: (Int) => Boolean): Unit = { pruned = true }
}
