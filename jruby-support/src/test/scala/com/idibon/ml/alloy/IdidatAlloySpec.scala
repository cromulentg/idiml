package com.idibon.ml.alloy

import scala.collection.JavaConverters._

import com.idibon.ml.common.{EmbeddedEngine, Engine, Archivable, ArchiveLoader}
import com.idibon.ml.predict.rules.DocumentRules
import com.idibon.ml.predict.ensemble.GangModel
import com.idibon.ml.predict._
import com.idibon.ml.feature._

import org.jruby.Ruby
import org.jruby.java.proxies.JavaProxy
import org.scalatest.{Matchers, FunSpec}
import org.json4s.JsonDSL._
import org.json4s._

import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.linalg.Vectors


class IdidatAlloySpec extends FunSpec with Matchers {

  def num(x: Float) = java.lang.Float.valueOf(x).asInstanceOf[java.lang.Number]

  describe("loadClassificationTask") {

    val labelFoo = new Label("00000000-0000-0000-0000-000000000001", "foo")

    val fooRules = new DocumentRules(labelFoo.uuid.toString,
      List("hello" -> 1.0f))

    it("should create alloys from nothing") {
      val alloy = IdidatAlloy.loadClassificationTask(new EmbeddedEngine,
        "testing", null, List(labelFoo).asJava,
        Map(labelFoo -> Map("foo" -> num(1.0f),
          "bar" -> num(0.0f)).asJava).asJava, false)

      val results = alloy.predict("content" -> "foobar", PredictOptions.DEFAULT)
      results shouldBe Seq(Classification(labelFoo.uuid.toString, 0.5f, 2, 3, Seq())).asJava
    }

    it("should replace existing rules models at the top level") {
      val initialAlloy = new BaseAlloy("testing",
        Seq(labelFoo), Map("foo" -> fooRules))
      val archive = scala.collection.mutable.HashMap[String, Array[Byte]]()
      initialAlloy.save(new MemoryAlloyWriter(archive))
      val updatedAlloy = IdidatAlloy.loadClassificationTask(new EmbeddedEngine,
        "updated", new MemoryAlloyReader(archive.toMap), List(labelFoo).asJava,
        Map(labelFoo -> Map("world" -> num(0.8f)).asJava).asJava, true)

      val r1 = updatedAlloy.predict("content" -> "hello", PredictOptions.DEFAULT)
      r1 shouldBe Seq(Classification(labelFoo.uuid.toString, 0.0f, 0, 2, Seq())).asJava
      val r2 = updatedAlloy.predict("content" -> "hello world", PredictOptions.DEFAULT)
      r2 shouldBe Seq(Classification(labelFoo.uuid.toString, 0.8f, 1, 2, Seq())).asJava
    }

    it("should replace rules models within gang models") {
      val innerGang = new GangModel(Map("inner" -> fooRules))
      val outerGang = new GangModel(Map("outer" -> innerGang))
      val initialAlloy = new BaseAlloy("testing",
        Seq(labelFoo), Map("gang" -> outerGang))
      val archive = scala.collection.mutable.HashMap[String, Array[Byte]]()
      initialAlloy.save(new MemoryAlloyWriter(archive))
      val updatedAlloy = IdidatAlloy.loadClassificationTask(new EmbeddedEngine,
        "updated", new MemoryAlloyReader(archive.toMap), List(labelFoo).asJava,
        Map(labelFoo -> Map("world" -> num(0.8f)).asJava).asJava, true)

      val r1 = updatedAlloy.predict("content" -> "hello", PredictOptions.DEFAULT)
      r1 shouldBe Seq(Classification(labelFoo.uuid.toString, 0.0f, 0, 2, Seq())).asJava
      val r2 = updatedAlloy.predict("content" -> "hello world", PredictOptions.DEFAULT)
      r2 shouldBe Seq(Classification(labelFoo.uuid.toString, 0.8f, 1, 2, Seq())).asJava
    }

    it("should keep other classification models") {
      val pipeline = (FeaturePipelineBuilder.named("pipeline")
        += FeaturePipelineBuilder.entry("dummy", new DummyTransformer, "$document")
        := "dummy").prime(Seq())
      val classifier = new VClassificationModel(labelFoo.uuid.toString)
      val innerGang = new GangModel(Map("inner" -> classifier))
      val outerGang = new GangModel(Map("outer" -> innerGang), Some(pipeline))
      val alloy = new BaseAlloy("testing", Seq(labelFoo), Map("gang" -> outerGang))
      val r1 = alloy.predict("content" -> "beta", PredictOptions.DEFAULT)
      r1 shouldBe Seq(Classification(labelFoo.uuid.toString, 0.25f, 1, 0, Seq())).asJava
      val archive = scala.collection.mutable.HashMap[String, Array[Byte]]()
      alloy.save(new MemoryAlloyWriter(archive))

      val updatedAlloy = IdidatAlloy.loadClassificationTask(new EmbeddedEngine,
        "updated", new MemoryAlloyReader(archive.toMap), List(labelFoo).asJava,
        Map().asJava, false)

      val r2 = updatedAlloy.predict("content" -> "beta", PredictOptions.DEFAULT)
      r2 shouldBe r1
    }
  }
}

class DummyTransformer extends FeatureTransformer
    with TerminableTransformer {

  def freeze { }

  def numDimensions = Some(1)

  def prune(pruneFn: (Int) => Boolean) { }

  def getFeatureByIndex(dim: Int): Option[Feature[_]] = None

  def apply(document: JObject): Vector = {
    val content = (document \ "content").asInstanceOf[JString].s
    Vectors.dense(1.0 / content.length)
  }
}

class VClassificationModel(label: String)
    extends PredictModel[Classification]
    with Archivable[VClassificationModel, VClassificationModelLoader] {

  val reifiedType = classOf[VClassificationModel]

  def predict(document: Document, options: PredictOptions) = {
    document.transformed.map({ case (vector, sigFeatFn) => {
      Seq(Classification(label, vector(0).toFloat, 1, 0, Seq()))
    }}).getOrElse({
      throw new UnsupportedOperationException("No pipeline")
    })
  }

  def getFeaturesUsed: Vector = ???

  def getEvaluationMetric: Double = ???

  def save(w: Alloy.Writer): Option[JObject] = {
    Some("label" -> label)
  }
}

class VClassificationModelLoader
    extends ArchiveLoader[VClassificationModel] {

  def load(engine: Engine, reader: Option[Alloy.Reader],
    config: Option[JObject]) = {

    val label = (config.get \ "label").asInstanceOf[JString].s
    new VClassificationModel(label)
  }
}
