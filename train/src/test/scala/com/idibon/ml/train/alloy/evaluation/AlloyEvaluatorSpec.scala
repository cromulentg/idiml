package com.idibon.ml.train.alloy.evaluation

import java.util

import com.idibon.ml.alloy.Alloy
import com.idibon.ml.common.EmbeddedEngine
import com.idibon.ml.feature._
import com.idibon.ml.feature.tokenizer.{Tag, Token}
import com.idibon.ml.predict.crf.BIOType
import com.idibon.ml.predict.ml.metrics._
import com.idibon.ml.predict.{Classification, Label, PredictOptions, Span}
import com.idibon.ml.train.alloy.{DataSetInfo, TrainingDataSet, DummyAlloyEvaluator}
import com.idibon.ml.train.datagenerator.crf.BIOTagger
import org.json4s.JsonAST.JObject
import org.json4s.native.JsonMethods._
import org.scalatest._

import scala.collection.JavaConversions._
import scala.collection.JavaConverters._


/**
  * Tests the Alloy Evaluator classes
  */
class AlloyEvaluatorSpec extends FunSpec
  with Matchers with BeforeAndAfter with ParallelTestExecution with BeforeAndAfterAll {
  implicit val formats = org.json4s.DefaultFormats
  val engine = new EmbeddedEngine

  describe("MultiClassMetricsEvaluator tests") {
    it("createEvaluationDataPoint works as expected"){
      val me = new MultiClassMetricsEvaluator(null, 0.5f)
      val classifications = List(
        new Classification("m", 0.35f, 1, 0, Seq()),
        new Classification("n", 0.3f, 1, 0, Seq())
      ).asJava
      val Some(actual) = me.createEvaluationDataPoint(
        Map("m" -> 0.0, "n" -> 1.0), Map("m"->Seq(EvaluationAnnotation(LabelName("m"), true, None, None))),
        classifications, Map("m" -> 0.5f))
      val expected = (Array(0.0), Array(0.0))
      actual.predicted(0) shouldBe expected._1(0)
      actual.gold(0) shouldBe expected._2(0)
    }
    it("getMaxLabel something over default threshold, no thresholds passed") {
      val me = new MultiClassMetricsEvaluator(null, 0.5f)
      val classifications = List(
        new Classification("m", 0.35f, 1, 0, Seq()),
        new Classification("n", 0.6f, 1, 0, Seq())
      ).asJava
      val actual = me.getMaxLabel(classifications, Map())
      actual shouldBe classifications(1)
    }
    it("getMaxLabel something over threshold, with thresholds passed") {
      val me = new MultiClassMetricsEvaluator(null, 0.5f)
      val classifications = List(
        new Classification("m", 0.35f, 1, 0, Seq()),
        new Classification("n", 0.6f, 1, 0, Seq())
      ).asJava
      val actual = me.getMaxLabel(classifications, Map("m" -> 0.3f, "n" -> 0.7f))
      actual shouldBe classifications(0)
    }
    it("getMaxLabel none over any threshold") {
      val me = new MultiClassMetricsEvaluator(null, 0.7f)
      val classifications = List(
        new Classification("m", 0.35f, 1, 0, Seq()),
        new Classification("n", 0.6f, 1, 0, Seq())
      ).asJava
      val actual = me.getMaxLabel(classifications, Map("m" -> 0.7f))
      actual shouldBe classifications(1)
    }
    it("createTrainingSummary works as expected") {
      val me = new MultiClassMetricsEvaluator(null, 0.7f)
      val Seq(actual) = me.createTrainingSummary(engine, Seq(
                new ClassificationEvaluationDataPoint(Array(1.0), Array(1.0), Seq()),
                new ClassificationEvaluationDataPoint(Array(0.0), Array(0.0), Seq())
              ), Map("a" -> 1.0, "b" -> 0.0), "name")
      actual.identifier shouldBe "name"
      actual.metrics.size shouldBe 18
      val expected = Some(new FloatMetric(MetricTypes.F1, MetricClass.Multiclass, 1.0f))
      actual.metrics.find(m => m.metricType == MetricTypes.F1) shouldBe expected
    }
  }

  describe("MultiLabelMetricsEvaluator tests") {
    it("createEvaluationDataPoint some over default threshold"){
      val me = new MultiLabelMetricsEvaluator(null, 0.2f)
      val classifications = List(
        new Classification("m", 0.35f, 1, 0, Seq()),
        new Classification("n", 0.3f, 1, 0, Seq())
      ).asJava
      val Some(actual) = me.createEvaluationDataPoint(
        Map("m" -> 0.0, "n" -> 1.0),
        Map("m"->Seq(EvaluationAnnotation(LabelName("m"), true, None, None)),
          "n"->Seq(EvaluationAnnotation(LabelName("n"), true, None, None))),
        classifications, Map("m" -> 0.5f))
      val expected = (Array(1.0), Array(0.0, 1.0))
      actual.predicted(0) shouldBe expected._1(0)
      actual.gold(0) shouldBe expected._2(0)
      actual.gold(1) shouldBe expected._2(1)
    }
    it("createEvaluationDataPoint some over passed in threshold"){
      val me = new MultiLabelMetricsEvaluator(null, 0.5f)
      val classifications = List(
        new Classification("m", 0.35f, 1, 0, Seq()),
        new Classification("n", 0.3f, 1, 0, Seq())
      ).asJava
      val Some(actual) = me.createEvaluationDataPoint(
        Map("m" -> 0.0, "n" -> 1.0),
        Map("m"->Seq(EvaluationAnnotation(LabelName("m"), true, None, None)),
          "n"->Seq(EvaluationAnnotation(LabelName("n"), true, None, None))),
        classifications, Map("m" -> 0.3f))
      val expected = (Array(0.0), Array(0.0, 1.0))
      actual.predicted(0) shouldBe expected._1(0)
      actual.gold(0) shouldBe expected._2(0)
      actual.gold(1) shouldBe expected._2(1)
    }
    it("createEvaluationDataPoint none over threshold"){
      val me = new MultiLabelMetricsEvaluator(null, 0.7f)
      val classifications = List(
        new Classification("m", 0.35f, 1, 0, Seq()),
        new Classification("n", 0.3f, 1, 0, Seq())
      ).asJava
      val Some(actual) = me.createEvaluationDataPoint(
        Map("m" -> 0.0, "n" -> 1.0),
        Map("m"->Seq(EvaluationAnnotation(LabelName("m"), true, None, None)),
          "n"->Seq(EvaluationAnnotation(LabelName("n"), true, None, None))
        ), classifications, Map("m" -> 0.5f))
      val expected = (Array(), Array(0.0, 1.0))
      actual.predicted.length shouldBe expected._1.length
      actual.gold(0) shouldBe expected._2(0)
      actual.gold(1) shouldBe expected._2(1)
    }
    it("createTrainingSummary works as expected") {
      val me = new MultiLabelMetricsEvaluator(null, 0.7f)
      val Seq(actual) = me.createTrainingSummary(engine, Seq(
                new ClassificationEvaluationDataPoint(Array(1.0), Array(1.0), Seq()),
                new ClassificationEvaluationDataPoint(Array(0.0), Array(0.0), Seq())
              ), Map("a" -> 1.0, "b" -> 0.0), "name")
      actual.identifier shouldBe "name"
      actual.metrics.size shouldBe 17
      val expected = Some(new FloatMetric(MetricTypes.F1, MetricClass.Multilabel, 1.0f))
      actual.metrics.find(m => m.metricType == MetricTypes.F1) shouldBe expected
    }
  }

  describe("AlloyEvaluatorSpec tests") {
    it("createPerLabelMetricsFromProbabilities correctly") {
      val edps = Seq(
        new ClassificationEvaluationDataPoint(Array(0.0), Array(1.0), Seq((0.0, 0.2f), (1.0, 0.3f))),
        new ClassificationEvaluationDataPoint(Array(1.0), Array(0.0), Seq((0.0, 0.3f), (1.0, 0.4f)))
      )
      val ltodb = Map("a" -> 0.0, "b" -> 1.0)
      val x = new DummyAlloyEvaluator()
      val actual = x.createPerLabelMetricsFromProbabilities(engine, ltodb, edps, MetricClass.Binary)
      actual.sortBy(x => x match {
        case a: LabelFloatListMetric => (a.label, a.metricType)
        case b: LabelFloatMetric => (b.label, b.metricType)
      }) shouldBe Seq[Metric with Buildable[_, _]](
        new LabelFloatMetric(
          MetricTypes.LabelBestF1Threshold, MetricClass.Binary, "a", 0.3f),
        new LabelFloatListMetric(
          MetricTypes.LabelProbabilities, MetricClass.Binary, "a", Seq(0.2f, 0.3f)),
        new LabelFloatMetric(
          MetricTypes.LabelBestF1Threshold, MetricClass.Binary, "b", 0.3f),
      new LabelFloatListMetric(
            MetricTypes.LabelProbabilities, MetricClass.Binary, "b", Seq(0.3f, 0.4f))
      )
    }
    it("collatePerLabelProbabilities correctly") {
      val dps = Seq((0.0, 0.2f), (1.0, 0.3f), (0.0, 0.3f), (1.0, 0.4f))
      val dbtl = Map(0.0 -> "a", 1.0 -> "b")
      val x = new DummyAlloyEvaluator()
      val actual = x.collatePerLabelProbabilities(dps, dbtl, MetricClass.Alloy)
      actual.sortBy(x => x.label) shouldBe Seq(
        new LabelFloatListMetric(
          MetricTypes.LabelProbabilities, MetricClass.Alloy, "a", Seq(0.2f, 0.3f)),
        new LabelFloatListMetric(
          MetricTypes.LabelProbabilities, MetricClass.Alloy, "b", Seq(0.3f, 0.4f))
      )
    }
    it("creates best label f1 metrics properly") {
      val edps = Seq(
        new ClassificationEvaluationDataPoint(Array(0.0), Array(1.0), Seq((0.0, 0.4f), (1.0, 0.3f))),
        new ClassificationEvaluationDataPoint(Array(1.0), Array(0.0), Seq((0.0, 0.3f), (1.0, 0.4f))),
        new ClassificationEvaluationDataPoint(Array(0.0), Array(0.0), Seq((0.0, 0.6f), (1.0, 0.4f))),
        new ClassificationEvaluationDataPoint(Array(1.0), Array(1.0), Seq((0.0, 0.3f), (1.0, 0.5f))),
        new ClassificationEvaluationDataPoint(Array(1.0), Array(1.0), Seq((0.0, 0.35f), (1.0, 0.56f)))
      )
      val dbletol = Map(0.0 -> "a", 1.0 -> "b")
      val x = new DummyAlloyEvaluator()
      val actual = x.getSuggestedLabelThreshold(engine, edps, dbletol, MetricClass.Alloy)
      val expected = Seq(
        new LabelFloatMetric(MetricTypes.LabelBestF1Threshold, MetricClass.Alloy, "a", 0.6f),
        new LabelFloatMetric(MetricTypes.LabelBestF1Threshold, MetricClass.Alloy, "b", 0.5f)
      )
      actual.sortBy(x => x.label) shouldBe expected
    }

    it("computes computeBestF1Threshold correctly") {
      val sqlContext = new org.apache.spark.sql.SQLContext(engine.sparkContext)
      val x = new DummyAlloyEvaluator()
      val values = Seq((0.5, 1.0), (0.4, 0.0), (0.3, 0.0), (0.5, 1.0), (0.6, 1.0))
      val actual = x.computeBestF1Threshold(engine, sqlContext, values)
      actual shouldBe 0.5f
    }

    it("handles degenerate all negatives without breaking") {
      val e = new MultiClassMetricsEvaluator(engine, 0f)
      val allNegs =
        parse("""
          {"content": "", "annotations": [{"label": {"name": "a"}, "isPositive": false},
           {"label": {"name": "b"}, "isPositive": false}]}
              """)
      val actual = e.getGoldSet(allNegs.extract[JObject])
      actual shouldBe Map[String, EvaluationAnnotation]()
    }
    it("handles mutli label data data") {
      val e = new MultiClassMetricsEvaluator(engine, 0f)
      val allNegs =
        parse("""
            {"content": "", "annotations": [
            {"label": {"name": "a"}, "isPositive": true},
            {"label": {"name": "b"}, "isPositive": false},
            {"label": {"name": "c"}, "isPositive": true},
            {"label": {"name": "d"}, "isPositive": false},
          ]}""")
      val actual = e.getGoldSet(allNegs.extract[JObject])
      actual shouldBe Map[String, Seq[EvaluationAnnotation]](
        "a" -> Seq(EvaluationAnnotation(LabelName("a"), true, None, None)),
        "c" -> Seq(EvaluationAnnotation(LabelName("c"), true, None, None)))
    }
    it("handles mutually exclusive label data data") {
      val e = new MultiClassMetricsEvaluator(engine, 0f)
      val allNegs =
        parse("""
          {"content": "", "annotations": [
            {"label": {"name": "a"}, "isPositive": false},
            {"label": {"name": "b"}, "isPositive": false},
            {"label": {"name": "c"}, "isPositive": false},
            {"label": {"name": "d"}, "isPositive": true},
          ]}""")
      val actual = e.getGoldSet(allNegs.extract[JObject])
      actual shouldBe Map[String, Seq[EvaluationAnnotation]](
        "d" -> Seq(EvaluationAnnotation(LabelName("d"), true, None, None)))
    }

    it("generates evaluation points handles empty gold set") {
      val onePos =
        parse("""
          {"content": "", "annotations": [
            {"label": {"name": "00000000-0000-0000-0000-000000000000"}, "isPositive": false},
            {"label": {"name": "00000000-0000-0000-0000-000000000001"}, "isPositive": false},
            {"label": {"name": "00000000-0000-0000-0000-000000000002"}, "isPositive": false},
            {"label": {"name": "00000000-0000-0000-0000-000000000003"}, "isPositive": false},
          ]}""").extract[JObject]
      val labels2Dble = Map("00000000-0000-0000-0000-000000000000" -> 0.0,
        "00000000-0000-0000-0000-000000000001"-> 1.0,
        "00000000-0000-0000-0000-000000000002" -> 2.0,
        "00000000-0000-0000-0000-000000000003" -> 3.0)
      val e = new MultiClassMetricsEvaluator(engine, 0f)
      val da = new Dummy2Alloy()
      val actual = e.generateEvaluationPoints(() => Seq(onePos), da, labels2Dble, Map())
      actual shouldBe Seq()
    }

    it("generates evaluation points correctly") {
      val onePos =
        parse("""
          {"content": "", "annotations": [
            {"label": {"name": "00000000-0000-0000-0000-000000000000"}, "isPositive": false},
            {"label": {"name": "00000000-0000-0000-0000-000000000001"}, "isPositive": false},
            {"label": {"name": "00000000-0000-0000-0000-000000000002"}, "isPositive": false},
            {"label": {"name": "00000000-0000-0000-0000-000000000003"}, "isPositive": true},
          ]}""").extract[JObject]
      val labels2Dble = Map("00000000-0000-0000-0000-000000000000" -> 0.0,
        "00000000-0000-0000-0000-000000000001"-> 1.0,
        "00000000-0000-0000-0000-000000000002" -> 2.0,
        "00000000-0000-0000-0000-000000000003" -> 3.0)
      val e = new MultiClassMetricsEvaluator(engine, 0f)
      val da = new Dummy2Alloy()
      val actual = e.generateEvaluationPoints(() => Seq(onePos), da, labels2Dble, Map())
      val expected = new ClassificationEvaluationDataPoint(Array(0.0), Array(3.0), Seq((0.0, 0.5f)))
      actual.size shouldBe 1
      actual(0).gold shouldBe expected.gold
      actual(0).predicted shouldBe expected.predicted
      actual(0).rawProbabilities shouldBe expected.rawProbabilities
    }

    it("evaluates an alloy properly") {
      val onePos =
        parse("""
          {"content": "", "annotations": [
            {"label": {"name": "00000000-0000-0000-0000-000000000000"}, "isPositive": false},
            {"label": {"name": "00000000-0000-0000-0000-000000000001"}, "isPositive": false},
            {"label": {"name": "00000000-0000-0000-0000-000000000002"}, "isPositive": false},
            {"label": {"name": "00000000-0000-0000-0000-000000000003"}, "isPositive": true},
          ]}""").extract[JObject]
      val labels2Dble = Map(new Label("00000000-0000-0000-0000-000000000000", "a") -> 0.0,
        new Label("00000000-0000-0000-0000-000000000001", "b") -> 1.0,
        new Label("00000000-0000-0000-0000-000000000002", "c") -> 2.0,
        new Label("00000000-0000-0000-0000-000000000003", "d") -> 3.0)
      val ds = new TrainingDataSet(
        new DataSetInfo(0, 0.0, labels2Dble), () => Seq(onePos))
      val e = new MultiClassMetricsEvaluator(engine, 0f)
      val da = new Dummy2Alloy()
      val summaries = e.evaluate("monkey", da, ds)
      summaries.size shouldBe 1
      summaries(0).metrics.size shouldBe 15
      val granularity = summaries(0).metrics.filter(x => x.metricType == MetricTypes.Notes).head
      val granString = granularity
        .asInstanceOf[PropertyMetric]
        .properties
        .filter(x => x._1.equals(AlloyEvaluator.GRANULARITY)).head._2
      granString shouldBe "Document"
    }
  }

  describe("BIOSpanMetricsEvaluator tests") {
    var bIOTagger: JunkBIOTagger = null

    before {
      bIOTagger = new JunkBIOTagger()
    }
    it("gold set creation creates proper EvaluationAnnotations") {
      val onePos =
        parse("""
          {"content": "\n   John Paul Walsh, an engineer who
          was the scientist at Cape Canaveral in charge of the launching of the second U.S.
          satellite\nto go into orbit.",
          "metadata":{"iso_639_1":"en"},
          "annotations":[
          {"label":{"name":"8a3b40db-4502-48e8-a675-d3ef1c03c74e"},"isPositive":true,"offset":4,"length":13},
          {"label":{"name":"6f5a5c01-4888-4792-af14-79d29ce9ff60"},"isPositive":true,"offset":24,"length":8},
          {"label":{"name":"6f5a5c01-4888-4792-af14-79d29ce9ff60"},"isPositive":true,"offset":55,"length":9},
          {"label":{"name":"666f5d3e-5d3e-4f0c-b0d7-fd85ee835b6c"},"isPositive":true,"offset":68,"length":13},
          {"label":{"name":"666f5d3e-5d3e-4f0c-b0d7-fd85ee835b6c"},"isPositive":true,"offset":124,"length":3}
          ]}""").extract[JObject]
      val e = new BIOSpanMetricsEvaluator(engine, bIOTagger)
      val actual = e.getGoldSet(onePos)
      val expected = Map(
        "8a3b40db-4502-48e8-a675-d3ef1c03c74e" -> Seq(
          EvaluationAnnotation(LabelName("8a3b40db-4502-48e8-a675-d3ef1c03c74e"), true, Some(4), Some(13),
            Some(List(Token("John", Tag.Word, 4, 4), Token("Paul", Tag.Word, 9, 4), Token("Walsh", Tag.Word, 14, 5))),
            Some(List(BIOType.BEGIN, BIOType.INSIDE, BIOType.INSIDE)))),
        "6f5a5c01-4888-4792-af14-79d29ce9ff60" -> Seq(
          EvaluationAnnotation(LabelName("6f5a5c01-4888-4792-af14-79d29ce9ff60"), true, Some(24), Some(8),
            Some(List(Token("engineer", Tag.Word, 24, 8))), Some(List(BIOType.BEGIN))),
          EvaluationAnnotation(LabelName("6f5a5c01-4888-4792-af14-79d29ce9ff60"), true, Some(55), Some(9),
            Some(List(Token("scientist", Tag.Word, 55, 9))), Some(List(BIOType.BEGIN)))),
        "666f5d3e-5d3e-4f0c-b0d7-fd85ee835b6c" -> Seq(
          EvaluationAnnotation(LabelName("666f5d3e-5d3e-4f0c-b0d7-fd85ee835b6c"), true, Some(68), Some(13),
            Some(List(Token("Cape", Tag.Word, 68, 4), Token("Canaveral", Tag.Word, 73, 9))), Some(List(BIOType.BEGIN, BIOType.INSIDE))),
          EvaluationAnnotation(LabelName("666f5d3e-5d3e-4f0c-b0d7-fd85ee835b6c"), true, Some(124), Some(3),
            Some(List(Token("U.S", Tag.Word, 124, 3))), Some(List(BIOType.BEGIN))))
      )
      actual.foreach({case (key, value) =>
        value.sortBy(x => x.offset) shouldBe expected(key)
      })
      actual.size shouldBe expected.size
    }

    it("gold set creation filters negative EvaluationAnnotations & zero length EvaluationAnnotations") {
      val onePos =
        parse("""
          {"content": "\n   John Paul Walsh, an engineer who
          was the scientist at Cape Canaveral in charge of the launching of the second U.S.
          satellite\nto go into orbit.",
          "metadata":{"iso_639_1":"en"},
          "annotations":[
          {"label":{"name":"8a3b40db-4502-48e8-a675-d3ef1c03c74e"},"isPositive":false,"offset":4,"length":13},
          {"label":{"name":"666f5d3e-5d3e-4f0c-b0d7-fd85ee835b6c"},"isPositive":true,"offset":-68,"length":-13},
          {"label":{"name":"666f5d3e-5d3e-4f0c-b0d7-fd85ee835b6c"},"isPositive":false,"offset":-124,"length":0}
          ]}""").extract[JObject]
      val e = new BIOSpanMetricsEvaluator(engine, bIOTagger)
      val actual = e.getGoldSet(onePos)
      actual shouldBe Map()
    }

    it("creates createEvaluationDataPoint correctly") {
      val onePos =
        parse("""
          {"content": "\n   John Paul Walsh, an engineer",
          "metadata":{"iso_639_1":"en"},
          "annotations":[
          {"label":{"name":"8a3b40db-4502-48e8-a675-d3ef1c03c74e"},"isPositive":true,"offset":4,"length":13},
          {"label":{"name":"6f5a5c01-4888-4792-af14-79d29ce9ff60"},"isPositive":true,"offset":24,"length":8},
          ]}""").extract[JObject]
      val e = new BIOSpanMetricsEvaluator(engine, bIOTagger)
      val l2d = Map("8a3b40db-4502-48e8-a675-d3ef1c03c74e" -> 0.0, "6f5a5c01-4888-4792-af14-79d29ce9ff60" -> 1.0)
      val gs = e.getGoldSet(onePos)
      val c = List(Span("6f5a5c01-4888-4792-af14-79d29ce9ff60", 0.5f, 0, 24, 8, Seq(Token("engineer", Tag.Word, 24, 8)),
        Seq(BIOType.BEGIN))).asJava
      val Some(actual) = e.createEvaluationDataPoint(l2d, gs, c, Map())
      val expected = SpanEvaluationDataPoint(
        Array(1.0, 1.0), Array(0.0, 1.0, 0.0, 1.0), Seq((1.0, 0.5f)),
        Seq(TokenDataPoint(0.0, 0, 0, 3), TokenDataPoint(1.0, 1, 0, 0)),
        Seq(TokenTagDataPoint("BEGIN", 1, 0, 1), TokenTagDataPoint("INSIDE", 0, 0, 2)))
      actual.predicted shouldBe expected.predicted
      actual.gold shouldBe expected.gold
      actual.rawProbabilities shouldBe expected.rawProbabilities
      val spanActual = actual.asInstanceOf[SpanEvaluationDataPoint]
      spanActual.tokenDP shouldBe expected.tokenDP
      spanActual.tokenTagDP shouldBe expected.tokenTagDP
    }

    it("handles exact match data point creation in exactMatchEvalDataPoint") {
      val onePos =
        parse("""
          {"content": "\n   John Paul Walsh, an engineer",
          "metadata":{"iso_639_1":"en"},
          "annotations":[
          {"label":{"name":"8a3b40db-4502-48e8-a675-d3ef1c03c74e"},"isPositive":true,"offset":4,"length":13},
          {"label":{"name":"6f5a5c01-4888-4792-af14-79d29ce9ff60"},"isPositive":true,"offset":24,"length":8},
          ]}""").extract[JObject]
      val e = new BIOSpanMetricsEvaluator(engine, bIOTagger)
      val l2d = Map("8a3b40db-4502-48e8-a675-d3ef1c03c74e" -> 0.0, "6f5a5c01-4888-4792-af14-79d29ce9ff60" -> 1.0)
      val gs = e.getGoldSet(onePos)
      val s = Span("6f5a5c01-4888-4792-af14-79d29ce9ff60", 0.5f, 0, 24, 8,
        Seq(Token("engineer", Tag.Word, 24, 8)),
        Seq(BIOType.BEGIN))
      val actual = e.exactMatchEvalDataPoint(l2d, gs, Seq(s))
      val expected = ExactMatchCounts(
        PredictedMatchCounts(Seq(1.0), Seq(1)),
        GoldMatchCounts(Seq(0.0, 1.0), Seq(0, 1)),
        Seq(PredictedProbability(1.0, 0.5f))
      )
      actual shouldBe expected
    }

    it("handles wrong label in exact match data point creation in exactMatchEvalDataPoint") {
      val onePos =
        parse("""
          {"content": "\n   John Paul Walsh, an engineer",
          "metadata":{"iso_639_1":"en"},
          "annotations":[
          {"label":{"name":"a"},"isPositive":true,"offset":4,"length":13},
          {"label":{"name":"b"},"isPositive":true,"offset":24,"length":8},
          ]}""").extract[JObject]
      val e = new BIOSpanMetricsEvaluator(engine, bIOTagger)
      val l2d = Map("a" -> 0.0, "b" -> 1.0)
      val gs = e.getGoldSet(onePos)
      val s = Span("a", 0.5f, 0, 24, 8,
        Seq(Token("engineer", Tag.Word, 24, 8)),
        Seq(BIOType.BEGIN))
      val actual = e.exactMatchEvalDataPoint(l2d, gs, Seq(s))
      val expected = ExactMatchCounts(
        PredictedMatchCounts(Seq(0.0), Seq(0)),
        GoldMatchCounts(Seq(1.0, 0.0), Seq(0, 0)),
        Seq(PredictedProbability(0.0, 0.5f))
      )
      actual shouldBe expected
    }

    it("handles wrong length but matching label in exactMatchEvalDataPoint") {
      val onePos =
        parse("""
          {"content": "\n   John Paul Walsh, an engineer",
          "metadata":{"iso_639_1":"en"},
          "annotations":[
          {"label":{"name":"a"},"isPositive":true,"offset":4,"length":13},
          {"label":{"name":"b"},"isPositive":true,"offset":24,"length":8},
          ]}""").extract[JObject]
      val e = new BIOSpanMetricsEvaluator(engine, bIOTagger)
      val l2d = Map("a" -> 0.0, "b" -> 1.0)
      val gs = e.getGoldSet(onePos)
      val s = Span("b", 0.5f, 0, 24, 10,
        Seq(Token("engineer", Tag.Word, 24, 10)),
        Seq(BIOType.BEGIN))
      val actual = e.exactMatchEvalDataPoint(l2d, gs, Seq(s))
      val expected = ExactMatchCounts(
        PredictedMatchCounts(Seq(1.0), Seq(0)),
        GoldMatchCounts(Seq(1.0, 0.0), Seq(0, 0)),
        Seq(PredictedProbability(1.0, 0.5f))
      )
      actual shouldBe expected
    }

    it("makes sure number of gold is done correctly and not affected by bad flatmap in exactMatchEvalDataPoint") {
      val onePos =
        parse("""
          {"content": "\n   John Paul Walsh, an engineer, went to town.",
          "metadata":{"iso_639_1":"en"},
          "annotations":[
          {"label":{"name":"a"},"isPositive":true,"offset":4,"length":13},
          {"label":{"name":"b"},"isPositive":true,"offset":24,"length":8},
          {"label":{"name":"b"},"isPositive":true,"offset":35,"length":4},
          {"label":{"name":"b"},"isPositive":true,"offset":44,"length":4}
          ]}""").extract[JObject]
      val e = new BIOSpanMetricsEvaluator(engine, bIOTagger)
      val l2d = Map("a" -> 0.0, "b" -> 1.0)
      val gs = e.getGoldSet(onePos)
      val s = Span("b", 0.5f, 0, 24, 10,
        Seq(Token("engineer", Tag.Word, 24, 10)),
        Seq(BIOType.BEGIN))
      val actual = e.exactMatchEvalDataPoint(l2d, gs, Seq(s))
      val expected = ExactMatchCounts(
        PredictedMatchCounts(Seq(1.0), Seq(0)),
        GoldMatchCounts(Seq(1.0, 1.0, 1.0, 0.0), Seq(0, 0, 0, 0)),
        Seq(PredictedProbability(1.0, 0.5f))
      )
      actual shouldBe expected
    }

    it("handles exact token match in tokenEvalDataPoint") {
      val onePos =
        parse("""
          {"content": "\n   John Paul Walsh, an engineer",
          "metadata":{"iso_639_1":"en"},
          "annotations":[
          {"label":{"name":"a"},"isPositive":true,"offset":4,"length":13},
          {"label":{"name":"b"},"isPositive":true,"offset":24,"length":8},
          ]}""").extract[JObject]
      val e = new BIOSpanMetricsEvaluator(engine, bIOTagger)
      val l2d = Map("a" -> 0.0, "b" -> 1.0)
      val gs = e.getGoldSet(onePos)
      val s = Span("a", 0.5f, 0, 4, 13,
        Seq(Token("John", Tag.Word, 4, 4), Token("Paul", Tag.Word, 9, 4), Token("Walsh", Tag.Word, 14, 5)),
        Seq(BIOType.BEGIN, BIOType.INSIDE, BIOType.INSIDE))
      val actual = e.tokenEvalDataPoint(l2d, gs, Seq(s))
      actual shouldBe Seq(
        TokenDataPoint(0.0, 3, 0, 0), TokenDataPoint(1.0, 0, 0, 1)
      )
    }

    it("handles partial token match in tokenEvalDataPoint") {
      val onePos =
        parse("""
          {"content": "\n   John Paul Walsh, an engineer",
          "metadata":{"iso_639_1":"en"},
          "annotations":[
          {"label":{"name":"a"},"isPositive":true,"offset":4,"length":13},
          {"label":{"name":"b"},"isPositive":true,"offset":24,"length":8},
          ]}""").extract[JObject]
      val e = new BIOSpanMetricsEvaluator(engine, bIOTagger)
      val l2d = Map("a" -> 0.0, "b" -> 1.0)
      val gs = e.getGoldSet(onePos)
      val s1 = Span("a", 0.5f, 0, 4, 13,
        Seq(Token("John", Tag.Word, 4, 4), Token("Paul", Tag.Word, 9, 4)),
        Seq(BIOType.BEGIN, BIOType.INSIDE))
      val s2 = Span("b", 0.5f, 0, 4, 13, Seq(Token("Walsh", Tag.Word, 14, 5)), Seq(BIOType.BEGIN))
      val actual = e.tokenEvalDataPoint(l2d, gs, Seq(s1, s2))
      actual shouldBe Seq(
        TokenDataPoint(0.0, 2, 0, 1), TokenDataPoint(1.0, 0, 1, 1)
      )
    }

    it("handles less gold tokens but more prediction tokens in tokenEvalDataPoint") {
      val onePos =
        parse("""
          {"content": "\n   John Paul Walsh, an engineer",
          "metadata":{"iso_639_1":"en"},
          "annotations":[
          {"label":{"name":"a"},"isPositive":true,"offset":4,"length":4}
          ]}""").extract[JObject]
      val e = new BIOSpanMetricsEvaluator(engine, bIOTagger)
      val l2d = Map("a" -> 0.0, "b" -> 1.0)
      val gs = e.getGoldSet(onePos)
      val s1 = Span("a", 0.5f, 0, 4, 13,
        Seq(Token("John", Tag.Word, 4, 4), Token("Paul", Tag.Word, 9, 4),  Token("Walsh", Tag.Word, 14, 5)),
        Seq(BIOType.BEGIN, BIOType.INSIDE, BIOType.INSIDE))
      val actual = e.tokenEvalDataPoint(l2d, gs, Seq(s1))
      actual shouldBe Seq(
        TokenDataPoint(0.0, 1, 2, 0), TokenDataPoint(1.0, 0, 0, 0)
      )
    }

    it("handles empty gold and predicted for a label in tokenEvalDataPoint") {
      val l2d = Map("a" -> 0.0, "b" -> 1.0)
      val e = new BIOSpanMetricsEvaluator(engine, bIOTagger)
      val actual = e.tokenEvalDataPoint(l2d, Map("a" -> Seq(), "b" -> Seq()), Seq())
      actual shouldBe Seq(TokenDataPoint(0.0, 0, 0, 0), TokenDataPoint(1.0, 0, 0, 0))
    }

    it("handles same exact token tags in tokenTagEvalDataPoint") {
      val onePos =
        parse("""
          {"content": "\n   John Paul Walsh, an engineer",
          "metadata":{"iso_639_1":"en"},
          "annotations":[
          {"label":{"name":"a"},"isPositive":true,"offset":4,"length":13},
          {"label":{"name":"b"},"isPositive":true,"offset":24,"length":8},
          ]}""").extract[JObject]
      val e = new BIOSpanMetricsEvaluator(engine, bIOTagger)
      val l2d = Map("a" -> 0.0, "b" -> 1.0)
      val gs = e.getGoldSet(onePos)
      val s = Span("a", 0.5f, 0, 4, 13,
        Seq(Token("John", Tag.Word, 4, 4), Token("Paul", Tag.Word, 9, 4), Token("Walsh", Tag.Word, 14, 5)),
        Seq(BIOType.BEGIN, BIOType.INSIDE, BIOType.INSIDE))
      val actual = e.tokenTagEvalDataPoint(l2d, gs, Seq(s))
      actual.sortBy(x => x.tagName) shouldBe Seq(
        TokenTagDataPoint("BEGIN", 1, 0, 1), TokenTagDataPoint("INSIDE", 2, 0, 0)
      )
    }

    it("handles different token tags in tokenTagEvalDataPoint") {
      val onePos =
        parse("""
          {"content": "\n   John Paul Walsh, an engineer",
          "metadata":{"iso_639_1":"en"},
          "annotations":[
          {"label":{"name":"a"},"isPositive":true,"offset":4,"length":13},
          {"label":{"name":"b"},"isPositive":true,"offset":24,"length":8},
          ]}""").extract[JObject]
      val e = new BIOSpanMetricsEvaluator(engine, bIOTagger)
      val l2d = Map("a" -> 0.0, "b" -> 1.0)
      val gs = e.getGoldSet(onePos)
      val s = Span("a", 0.5f, 0, 4, 13,
        Seq(Token("John", Tag.Word, 4, 4), Token("Paul", Tag.Word, 9, 4), Token("Walsh", Tag.Word, 14, 5)),
        Seq(BIOType.INSIDE, BIOType.BEGIN, BIOType.BEGIN))
      val actual = e.tokenTagEvalDataPoint(l2d, gs, Seq(s))
      actual.sortBy(x => x.tagName) shouldBe Seq(
        TokenTagDataPoint("BEGIN", 0, 2, 2), TokenTagDataPoint("INSIDE", 0, 1, 2)
      )
    }

    it("handles differing lengths in gold and predicted in tokenTagEvalDataPoint") {
      val onePos =
        parse("""
          {"content": "\n   John Paul Walsh, an engineer",
          "metadata":{"iso_639_1":"en"},
          "annotations":[
          {"label":{"name":"a"},"isPositive":true,"offset":4,"length":13}
          ]}""").extract[JObject]
      val e = new BIOSpanMetricsEvaluator(engine, bIOTagger)
      val l2d = Map("a" -> 0.0, "b" -> 1.0)
      val gs = e.getGoldSet(onePos)
      val s = Span("a", 0.5f, 0, 4, 13,
        Seq(Token("John", Tag.Word, 4, 4)),
        Seq(BIOType.INSIDE))
      val actual = e.tokenTagEvalDataPoint(l2d, gs, Seq(s))
      actual.sortBy(x => x.tagName) shouldBe Seq(
        TokenTagDataPoint("BEGIN", 0, 0, 1), TokenTagDataPoint("INSIDE", 0, 1, 2)
      )
    }

    it("creates createTrainingSummary as expected") {
      val dps = Seq(
        new SpanEvaluationDataPoint(
          Array(0.0, 1.0, 0.0, 1.0), Array(0.0, 1.0, 0.0, 1.0),
          Seq((0.0, 0.5f), (1.0, 0.5f)),
          Seq(new TokenDataPoint(0.0, 3, 1, 0), new TokenDataPoint(1.0, 1, 3, 2)),
          Seq(new TokenTagDataPoint("a", 1, 3, 2), new TokenTagDataPoint("b", 1, 3, 2))
        ))
      val e = new BIOSpanMetricsEvaluator(engine, bIOTagger)
      val actual = e.createTrainingSummary(engine, dps, Map("a" -> 0.0, "b" -> 1.0), "test", 1.0)
      actual.size shouldBe 3
      actual.map(ts => {
        val Some(notes) = ts.metrics.find(p => p.metricType == MetricTypes.Notes)
        val granularity = notes.asInstanceOf[PropertyMetric].properties.head
        Granularity.withName(granularity._2)
      }).sortBy(x => x.id) shouldBe Seq(Granularity.Span, Granularity.Token, Granularity.TokenTag)
    }

    it("tokenTagMatchEval creates all metrics correctly") {
      val e = new BIOSpanMetricsEvaluator(engine, bIOTagger)
      val input = Seq(
        new TokenTagDataPoint("a", 1, 3, 2),
        new TokenTagDataPoint("a", 2, 2, 2),
        new TokenTagDataPoint("a", 3, 1, 0),
        new TokenTagDataPoint("b", 1, 3, 2),
        new TokenTagDataPoint("b", 2, 2, 2),
        new TokenTagDataPoint("b", 3, 1, 0)
      )
      val actual = e.tokenTagMatchEval(input).sortBy({
          case z: LabelFloatMetric => (z.metricType, z.label)
          case z: FloatMetric => (z.metricType, "")
      })
      val expected = Seq[Metric with Buildable[_, _]](
        new LabelFloatMetric(MetricTypes.LabelF1, MetricClass.Multiclass, "a", 0.54545456f),
        new LabelFloatMetric(MetricTypes.LabelF1, MetricClass.Multiclass, "b", 0.54545456f),
        new LabelFloatMetric(MetricTypes.LabelPrecision, MetricClass.Multiclass, "a", 0.5f),
        new LabelFloatMetric(MetricTypes.LabelPrecision, MetricClass.Multiclass, "b", 0.5f),
        new LabelFloatMetric(MetricTypes.LabelRecall, MetricClass.Multiclass, "a", 0.6f),
        new LabelFloatMetric(MetricTypes.LabelRecall, MetricClass.Multiclass, "b", 0.6f),
        new FloatMetric(MetricTypes.MacroF1, MetricClass.Multiclass, 0.54545456f),
        new FloatMetric(MetricTypes.MacroPrecision, MetricClass.Multiclass, 0.5f),
        new FloatMetric(MetricTypes.MacroRecall, MetricClass.Multiclass, 0.6f),
        new FloatMetric(MetricTypes.MicroF1, MetricClass.Multiclass, 0.54545456f),
        new FloatMetric(MetricTypes.MicroPrecision, MetricClass.Multiclass, 0.5f),
        new FloatMetric(MetricTypes.MicroRecall, MetricClass.Multiclass, 0.6f)
      )
      actual.size shouldBe expected.size
      actual shouldBe expected
    }

    it("sumGrouped sums numbers properly") {
      val e = new BIOSpanMetricsEvaluator(engine, bIOTagger)
      val input = Map("a" -> Seq(
        TokenDataPoint(0.0, 2, 2, 2), TokenDataPoint(0.0, 2, 2, 2)
      ), "b" -> Seq(
        TokenDataPoint(2.0, 2, 2, 2), TokenDataPoint(2.0, 2, 2, 2)
      ), "c" -> Seq(
        TokenDataPoint(1.0, 2, 2, 2), TokenDataPoint(1.0, 2, 2, 2)
      ))
      val actual = e.sumGrouped[String](input)
      val expected = Map(
        "a" -> (4, 4, 4),
        "b" -> (4, 4, 4),
        "c" -> (4, 4, 4)
      )
      actual shouldBe expected
    }

    it("filters numbers for computePRF1Metrics correctly") {
      val e = new BIOSpanMetricsEvaluator(engine, bIOTagger)
      val input = Map("a" -> (1, 2, 3), "b" -> (0, 0, 0), "c" -> (0, 0, 1))
      val actual = e.computePRF1Metrics[String](input)
      val expected = Map(
        "a" -> (1.0f/3.0f, 0.25f, 0.28571427f),
        "c" -> (0.0f, 0.0f, 0.0f)
      )
      actual shouldBe expected
    }

    it("createPRF1Metrics creates correctly") {
      val e = new BIOSpanMetricsEvaluator(engine, bIOTagger)
      val input = Map("a" -> (0.4f, 0.4f, 0.4f), "b" -> (0f, 0f, 0f))
      val actual = e.createPRF1Metrics(input, MetricClass.Alloy)
      val expected = Map(
        "a" -> (new LabelFloatMetric(MetricTypes.LabelPrecision, MetricClass.Alloy, "a", 0.4f),
                new LabelFloatMetric(MetricTypes.LabelRecall, MetricClass.Alloy, "a", 0.4f),
                new LabelFloatMetric(MetricTypes.LabelF1, MetricClass.Alloy, "a", 0.4f)),
        "b" -> (new LabelFloatMetric(MetricTypes.LabelPrecision, MetricClass.Alloy, "b", 0.0f),
                new LabelFloatMetric(MetricTypes.LabelRecall, MetricClass.Alloy, "b", 0.0f),
                new LabelFloatMetric(MetricTypes.LabelF1, MetricClass.Alloy, "b", 0.0f))
      )
      actual shouldBe expected
    }

    it("computeMicroMetrics evals correctly") {
      val e = new BIOSpanMetricsEvaluator(engine, bIOTagger)
      val input = Map("a" -> (1, 2, 3), "b" -> (0, 2, 0), "c" -> (0, 0, 1))
      val actual = e.computeMicroMetrics(input)
      val expected = Seq(
        new FloatMetric(MetricTypes.MicroPrecision, MetricClass.Multiclass, 0.2f),
        new FloatMetric(MetricTypes.MicroRecall, MetricClass.Multiclass, 0.2f),
        new FloatMetric(MetricTypes.MicroF1, MetricClass.Multiclass, 0.2f)
      )
      actual(0) shouldBe expected(0)
      actual(1) shouldBe expected(1)
      actual(2).float shouldBe (expected(2).float +- 0.0001f)
      actual(2).mType shouldBe expected(2).mType
      actual(2).mClass shouldBe expected(2).mClass
    }

    it("computeMacroMetrics evals correctly") {
      val e = new BIOSpanMetricsEvaluator(engine, bIOTagger)
      val input = Map(
        "a" -> (new LabelFloatMetric(MetricTypes.LabelPrecision, MetricClass.Alloy, "a", 0.4f),
          new LabelFloatMetric(MetricTypes.LabelRecall, MetricClass.Alloy, "a", 0.4f),
          new LabelFloatMetric(MetricTypes.LabelF1, MetricClass.Alloy, "a", 0.4f)),
        "b" -> (new LabelFloatMetric(MetricTypes.LabelPrecision, MetricClass.Alloy, "b", 0.0f),
          new LabelFloatMetric(MetricTypes.LabelRecall, MetricClass.Alloy, "b", 0.0f),
          new LabelFloatMetric(MetricTypes.LabelF1, MetricClass.Alloy, "b", 0.0f))
      )
      val actual = e.computeMacroMetrics(input, MetricClass.Alloy)
      val expected = Seq(
        new FloatMetric(MetricTypes.MacroPrecision, MetricClass.Alloy, 0.2f),
        new FloatMetric(MetricTypes.MacroRecall, MetricClass.Alloy, 0.2f),
        new FloatMetric(MetricTypes.MacroF1, MetricClass.Alloy, 0.2f)
      )
      actual shouldBe expected
    }

    it("tokenMatchEval evals correctly") {
      val e = new BIOSpanMetricsEvaluator(engine, bIOTagger)
      val input = Seq(
        new TokenDataPoint(0.0, 1, 3, 2),
        new TokenDataPoint(0.0, 2, 2, 2),
        new TokenDataPoint(0.0, 3, 1, 0),
        new TokenDataPoint(1.0, 1, 3, 2),
        new TokenDataPoint(1.0, 2, 2, 2),
        new TokenDataPoint(1.0, 3, 1, 0)
      )
      val actual = e.tokenMatchEval(input, Map("a" -> 0.0, "b" -> 1.0)).sortBy({
          case z: LabelFloatMetric => (z.metricType, z.label)
          case z: FloatMetric => (z.metricType, "")
      })
      val expected = Seq[Metric with Buildable[_, _]](
        new LabelFloatMetric(MetricTypes.LabelF1, MetricClass.Multiclass, "a", 0.54545456f),
        new LabelFloatMetric(MetricTypes.LabelF1, MetricClass.Multiclass, "b", 0.54545456f),
        new LabelFloatMetric(MetricTypes.LabelPrecision, MetricClass.Multiclass, "a", 0.5f),
        new LabelFloatMetric(MetricTypes.LabelPrecision, MetricClass.Multiclass, "b", 0.5f),
        new LabelFloatMetric(MetricTypes.LabelRecall, MetricClass.Multiclass, "a", 0.6f),
        new LabelFloatMetric(MetricTypes.LabelRecall, MetricClass.Multiclass, "b", 0.6f),
        new FloatMetric(MetricTypes.MacroF1, MetricClass.Multiclass, 0.54545456f),
        new FloatMetric(MetricTypes.MacroPrecision, MetricClass.Multiclass, 0.5f),
        new FloatMetric(MetricTypes.MacroRecall, MetricClass.Multiclass, 0.6f),
        new FloatMetric(MetricTypes.MicroF1, MetricClass.Multiclass, 0.54545456f),
        new FloatMetric(MetricTypes.MicroPrecision, MetricClass.Multiclass, 0.5f),
        new FloatMetric(MetricTypes.MicroRecall, MetricClass.Multiclass, 0.6f)
      )
      actual.size shouldBe expected.size
      actual shouldBe expected
    }

    it("exactMatchEval evaluates correctly") {
      val e = new BIOSpanMetricsEvaluator(engine, bIOTagger)
      val input = Seq(
        (Array(0.0, 1.0, 0.0, 1.0), Array(0.0, 1.0, 0.0, 1.0)),
        (Array(0.0, 1.0, 0.0, 0.0), Array(0.0, 1.0, 0.0, 0.0))
      )
      val actual = e.exactMatchEval(input, Map(0.0 -> "a", 1.0 -> "b")).sortBy({
          case z: LabelFloatMetric => (z.metricType, z.label)
          case z: FloatMetric => (z.metricType, "")
      })
      val expected = Seq[Metric with Buildable[_, _]](
        new LabelFloatMetric(MetricTypes.LabelF1, MetricClass.Multiclass, "a", 0.0f),
        new LabelFloatMetric(MetricTypes.LabelF1, MetricClass.Multiclass, "b", 0.5f),
        new LabelFloatMetric(MetricTypes.LabelPrecision, MetricClass.Multiclass, "a", 0.0f),
        new LabelFloatMetric(MetricTypes.LabelPrecision, MetricClass.Multiclass, "b", 0.5f),
        new LabelFloatMetric(MetricTypes.LabelRecall, MetricClass.Multiclass, "a", 0.0f),
        new LabelFloatMetric(MetricTypes.LabelRecall, MetricClass.Multiclass, "b", 0.5f),
        new FloatMetric(MetricTypes.MacroF1, MetricClass.Multiclass, 0.25f),
        new FloatMetric(MetricTypes.MacroPrecision, MetricClass.Multiclass, 0.25f),
        new FloatMetric(MetricTypes.MacroRecall, MetricClass.Multiclass, 0.25f),
        new FloatMetric(MetricTypes.MicroF1, MetricClass.Multiclass, 0.25f),
        new FloatMetric(MetricTypes.MicroPrecision, MetricClass.Multiclass, 0.25f),
        new FloatMetric(MetricTypes.MicroRecall, MetricClass.Multiclass, 0.25f)
      )
      actual.zip(expected).foreach({case (a1, e1) =>
        a1 shouldBe e1
      })
    }

    it("creates byLabelPrecisionRecallF1 correctly") {
      val e = new BIOSpanMetricsEvaluator(engine, bIOTagger)
      val input1 = Map(0.0 -> (2, 4), 1.0 -> (3, 4))
      val input2 = Map(0.0 -> (2, 5), 1.0 -> (3, 3))
      val actual = e.byLabelPrecisionRecallF1(input1, input2, Map(0.0 -> "a", 1.0 -> "b")).sortBy({
          case z: LabelFloatMetric => (z.metricType, z.label)
      })
      val expected = Seq(
        new LabelFloatMetric(MetricTypes.LabelF1, MetricClass.Multiclass, "a", 0.44444448f),
        new LabelFloatMetric(MetricTypes.LabelF1, MetricClass.Multiclass, "b", 0.85714287f),
        new LabelFloatMetric(MetricTypes.LabelPrecision, MetricClass.Multiclass, "a", 0.5f),
        new LabelFloatMetric(MetricTypes.LabelPrecision, MetricClass.Multiclass, "b", 0.75f),
        new LabelFloatMetric(MetricTypes.LabelRecall, MetricClass.Multiclass, "a", 0.4f),
        new LabelFloatMetric(MetricTypes.LabelRecall, MetricClass.Multiclass, "b", 1.0f)
      )
      actual shouldBe expected
    }

    it("creates microPrecisionRecallF1 correctly") {
      val e = new BIOSpanMetricsEvaluator(engine, bIOTagger)
      val input1 = Map(0.0 -> (2, 4), 1.0 -> (3, 4))
      val input2 = Map(0.0 -> (2, 5), 1.0 -> (3, 3))
      val actual = e.microPrecisionRecallF1(input1, input2, MetricClass.Alloy)
        .sortBy(x => x.metricType)
      val expected = Seq(
        new FloatMetric(MetricTypes.MicroF1, MetricClass.Alloy, 0.625f),
        new FloatMetric(MetricTypes.MicroPrecision, MetricClass.Alloy, 0.625f),
        new FloatMetric(MetricTypes.MicroRecall, MetricClass.Alloy, 0.625f)
      )
      actual shouldBe expected
    }

    it("computes F1 correctly") {
      val e = new BIOSpanMetricsEvaluator(engine, bIOTagger)
      e.computeF1(2.0f, 1.0f) shouldBe 1.3333334f
    }
    it("handles 0 precision and recall for f1 computation") {
      val e = new BIOSpanMetricsEvaluator(engine, bIOTagger)
      e.computeF1(0.0f, 0.0f) shouldBe 0.0f
    }

  }

  describe("NOOP tests") {
    it("doesn't break creating data points") {
      val n = new NoOpEvaluator()
      val Some(actual) = n.createEvaluationDataPoint(Map(), Map(), null, null)
      val expected = new ClassificationEvaluationDataPoint(Array(), Array(), Seq())
      actual.gold.length shouldBe expected.gold.length
      actual.predicted.length shouldBe expected.predicted.length
      actual.rawProbabilities.length shouldBe expected.rawProbabilities.length
    }
    it("doesn't break creating training summaries") {
      val n = new NoOpEvaluator()
      n.createTrainingSummary(null, Seq(), null, "", 0.0) shouldBe Seq()
    }
  }
}

class JunkBIOTagger() extends BIOTagger {
  var sequenceGenerator: SequenceGenerator = null
  var featureExtractor: ChainPipeline = null
  var bioTagger: BIOTagger = null

  def buildSequenceGenerator = (SequenceGeneratorBuilder("foo")
    += ("contentType", new contenttype.ContentTypeDetector(false), Seq("$document"))
    += ("lang", new language.LanguageDetector, Seq("$document", "contentType"))
    += ("content", new ContentExtractor, Seq("$document"))
    += ("tokenizer", new tokenizer.ChainTokenTransformer(Seq(tokenizer.Tag.Word)),
    Seq("content", "lang", "contentType"))
    := ("tokenizer"))

  def buildChainPipeline = (ChainPipelineBuilder("foo")
    += ("contentType", new contenttype.ContentTypeDetector(false), Seq("$document"))
    += ("lang", new language.LanguageDetector, Seq("$document", "contentType"))
    += ("words", new bagofwords.ChainBagOfWords(bagofwords.CaseTransform.ToLower),
    Seq("$sequence", "lang"))
    += ("lift", new ChainLiftTransformer(), Seq("words"))
    += ("index", new indexer.ChainIndexTransformer(), Seq("lift"))
    := ("index"))

  sequenceGenerator = buildSequenceGenerator
  featureExtractor = buildChainPipeline
}

//Dummy 2 since we already have a dummy alloy ...
class Dummy2Alloy extends Alloy[Classification] {
  override def getLabels: util.List[Label] = ???

  override def getSuggestedThresholds: util.Map[Label, java.lang.Float] = Map(
    new Label("00000000-0000-0000-0000-000000000000", "a") -> new java.lang.Float(0.4f),
    new Label("00000000-0000-0000-0000-000000000001", "b") -> new java.lang.Float(0.4f),
    new Label("00000000-0000-0000-0000-000000000002", "c") -> new java.lang.Float(0.4f),
    new Label("00000000-0000-0000-0000-000000000003", "d") -> new java.lang.Float(0.4f)
  ).asJava

  override def predict(document: JObject, options: PredictOptions): util.List[Classification] = {
    List(new Classification("00000000-0000-0000-0000-000000000000", 0.5f, 1, 0, Seq()))
  }

  override def save(writer: Alloy.Writer): Unit = ???

  override def translateUUID(uuid: String): Label = ???
}
