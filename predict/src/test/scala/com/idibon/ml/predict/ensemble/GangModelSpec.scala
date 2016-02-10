package com.idibon.ml.predict.ensemble

import scala.collection.mutable.HashMap

import com.idibon.ml.alloy.{Alloy, Codec, MemoryAlloyReader, MemoryAlloyWriter}
import com.idibon.ml.common.{Engine, ArchiveLoader, Archivable, EmbeddedEngine}
import com.idibon.ml.feature.indexer.IndexTransformer
import com.idibon.ml.feature.language.LanguageDetector
import com.idibon.ml.feature.tokenizer.TokenTransformer
import com.idibon.ml.feature._
import com.idibon.ml.predict.ml.{IdibonMultiClassLRModel, MLModel}
import com.idibon.ml.predict.rules.{RuleFeature, DocumentRules}
import com.idibon.ml.feature.bagofwords.Word
import com.idibon.ml.predict._
import org.apache.spark.mllib.classification.IdibonSparkMLLIBLRWrapper
import org.apache.spark.mllib.linalg.{Vectors, Vector}
import org.json4s._
import org.json4s.JsonDSL._
import org.scalatest.{BeforeAndAfter, FunSpec, Matchers}

/**
  * Class to test the Gang Model
  */
class GangModelSpec extends FunSpec with Matchers with BeforeAndAfter {

  val pipeline: FeaturePipeline = (FeaturePipelineBuilder.named("StefansPipeline")
    += (FeaturePipelineBuilder.entry("metaExtractor", new MetadataNumberExtractor, "$document"))
    := ("metaExtractor"))

  val text: String = "Everybody loves replacing hadoop with spark because it is much faster. a b d"
  val doc: JObject = JObject("content" -> text, "metadata" -> JObject(List(("number", JDouble(0.5)))))
  val fp = pipeline.prime(List(doc))

  describe("save and load") {
    it("should save and load properly") {
      val archive = HashMap[String, Array[Byte]]()
      val docRules1 = new DocumentRules("alabel", List(("loves", 0.5f)))
      val docRules2 = new DocumentRules("blabel", List(("is", 0.5f)))
      val mcModel = new IdibonMultiClassLRModel(Map("alabel" -> 0, "blabel" -> 1),
        new IdibonSparkMLLIBLRWrapper(Vectors.dense(Array(0.5, 0.5, 0.5)), 0.0, 3, 2), Some(fp))
      val gang1 = new GangModel(Map[String, PredictModel[Classification]](
        "0" -> mcModel, "1" -> docRules1, "2" -> docRules2))
      val metadata = gang1.save(new MemoryAlloyWriter(archive))
      val expectedMetadata = Some(JObject(List(
        ("labels", JArray(List(JString("0"), JString("1"), JString("2")))),
        ("model-meta", JObject(List(
          ("0",
            JObject(List(("config",
              JObject(List(("version",JString("0.0.2")),
                ("featurePipeline",JObject(List(
                  ("version",JString("0.0.1")),
                  ("transforms",
                    JArray(List(JObject(List(("name",JString("metaExtractor")),
                      ("class",JString("com.idibon.ml.predict.ensemble.MetadataNumberExtractor")),
                      ("config",JNothing)))))),
                  ("pipeline",
                    JArray(List(JObject(List(("name",JString("metaExtractor")),
                      ("inputs",JArray(List(JString("$document")))))),
                      JObject(List(("name",JString("$output")), ("inputs",JArray(List(JString("metaExtractor"))))))))))))))),
              ("class", JString("com.idibon.ml.predict.ml.IdibonMultiClassLRModel"))))),
          ("1", JObject(List(
            ("config", JObject(List(("label", JString("alabel"))))),
            ("class", JString("com.idibon.ml.predict.rules.DocumentRules"))))),
          ("2", JObject(List(
            ("config", JObject(List(("label", JString("blabel"))))),
            ("class", JString("com.idibon.ml.predict.rules.DocumentRules"))))
          )))),
        ("featurePipeline", JNothing)
      )))
      metadata shouldBe expectedMetadata
      val gang2 = (new GangModelLoader).load(
        new EmbeddedEngine, Some(new MemoryAlloyReader(archive.toMap)), metadata)
      val gang1Pred = gang1.predict(Document.document(doc),
        new PredictOptionsBuilder().showSignificantFeatures(0.4f).build())
      val gang2Pred = gang2.predict(Document.document(doc),
        new PredictOptionsBuilder().showSignificantFeatures(0.4f).build())
      gang1Pred shouldBe gang2Pred
    }

    it("should save and load empty models") {
      val archive = HashMap[String, Array[Byte]]()
      val docRules1 = new DocumentRules("alabel", List())
      val docRules2 = new DocumentRules("blabel", List())
      val gang1 = new GangModel(Map[String, PredictModel[Classification]](
        "0" -> new FakeMCModel(List("alabel", "blabel")), "1" -> docRules1, "2" -> docRules2))
      val metadata = gang1.save(new MemoryAlloyWriter(archive))
      val expectedMetadata = Some(JObject(List(
        ("labels", JArray(List(JString("0"), JString("1"), JString("2")))),
        ("model-meta",JObject(List(
          ("0", JObject(List(
            ("config", JNothing),
            ("class",JString("com.idibon.ml.predict.ensemble.FakeMCModel"))))),
          ("1", JObject(List(
            ("config", JObject(List(("label",JString("alabel"))))),
            ("class",JString("com.idibon.ml.predict.rules.DocumentRules"))))),
          ("2",JObject(List(
            ("config", JObject(List(("label",JString("blabel"))))),
            ("class",JString("com.idibon.ml.predict.rules.DocumentRules"))))
          )))),
        ("featurePipeline", JNothing))))
      metadata shouldBe expectedMetadata
      val gang2 = (new GangModelLoader).load(
        new EmbeddedEngine, Some(new MemoryAlloyReader(archive.toMap)), metadata)
      gang1 shouldBe gang2
    }

    it("should save and load with no per label models") {
      val archive = HashMap[String, Array[Byte]]()
      val gang1 = new GangModel(Map("0" -> new FakeMCModel(List("alabel", "blabel"))))
      val metadata = gang1.save(new MemoryAlloyWriter(archive))
      val expectedMetadata = Some(JObject(List(
        ("labels", JArray(List(JString("0")))),
        ("model-meta",JObject(List(
          ("0", JObject(List(
            ("config", JNothing),
            ("class",JString("com.idibon.ml.predict.ensemble.FakeMCModel")))))
        ))),
        ("featurePipeline", JNothing))))
      metadata shouldBe expectedMetadata
      val gang2 = (new GangModelLoader).load(
        new EmbeddedEngine, Some(new MemoryAlloyReader(archive.toMap)), metadata)
      gang1 shouldBe gang2
    }
  }

  describe("document prediction test cases") {
    it("should return results for all labels, even with no per label model") {
      val gang1 = new GangModel(Map[String, PredictModel[Classification]](
        "0" -> new FakeMCModel(List("alabel", "blabel"))))
      val doc = new JObject(List("content" -> new JString("string matching is working"),
        "metadata" -> JObject(List(("number", JDouble(0.5))))))
      val actual = gang1.predict(Document.document(doc), PredictOptions.DEFAULT)
      actual.map(_.label) shouldBe List("alabel", "blabel")
      actual.map(_.matchCount) shouldBe List(1, 1)
      actual.map(_.probability) shouldEqual List(0.2f, 0.2f)
      actual.map(_.significantFeatures) shouldEqual List(List(), List())
    }
    it("should combine results across models for each label, sorting by probability") {
      val docRules = new DocumentRules("blabel", List(("/str[ij]ng/", 0.6f), ("is", 0.6f)))
      val gang1 = new GangModel(Map[String, PredictModel[Classification]](
        "0" -> new FakeMCModel(List("alabel", "blabel")), "1" -> docRules))
      val doc = new JObject(List("content" -> new JString("string matching is working"),
        "metadata" -> JObject(List(("number", JDouble(0.5))))))
      val actual = gang1.predict(Document.document(doc), PredictOptions.DEFAULT)
      actual.map(_.label) shouldBe List("blabel", "alabel")
      actual.map(_.matchCount) shouldBe List(3, 1)
      actual.map(_.probability) shouldEqual List(0.4666667f, 0.2f)
      actual.map(_.significantFeatures) shouldEqual List(List(), List())
    }
    it("works as intended with white list & black list trigger in per label models") {
      val docRules1 = new DocumentRules("blabel", List(("/str[ij]ng/", 1.0f), ("is", 0.6f)))
      val docRules2 = new DocumentRules("alabel", List(("/str[ij]ng/", 0.0f), ("is", 0.6f)))
      val gang1 = new GangModel(Map[String, PredictModel[Classification]](
        "0" -> new FakeMCModel(List("alabel", "blabel")),
        "1" -> docRules1, "2" -> docRules2))
      val doc = new JObject(List("content" -> new JString("string matching is working"),
        "metadata" -> JObject(List(("number", JDouble(0.5))))))
      val actual = gang1.predict(Document.document(doc),
        new PredictOptionsBuilder().showSignificantFeatures(0.4f).build())
      actual.map(_.label) shouldBe List("blabel", "alabel")
      // since we whitelist/blacklist - we drop all the other model results, hence match count of 1.
      actual.map(_.matchCount) shouldBe List(1, 1)
      actual.map(_.probability) shouldEqual List(1.0f, 0.0f)
      actual.map(_.significantFeatures) shouldEqual List(
        List((RuleFeature("/str[ij]ng/"),1.0)), List((RuleFeature("/str[ij]ng/"),0.0)))
    }
    it("should return significant features correctly from all models") {
      val docRules1 = new DocumentRules("blabel", List(("/str[ij]ng/", 0.6f), ("is", 0.6f)))
      val docRules2 = new DocumentRules("alabel", List(("/str[ij]ng/", 0.3f), ("is", 0.8f)))
      val gang1 = new GangModel(Map[String, PredictModel[Classification]](
        "0" -> new FakeMCModel(List("alabel", "blabel")),
        "1" -> docRules1, "2" -> docRules2))
      val doc = new JObject(List("content" -> new JString("string matching is working"),
        "metadata" -> JObject(List(("number", JDouble(0.5))))))
      val actual = gang1.predict(Document.document(doc),
        new PredictOptionsBuilder().showSignificantFeatures(0.4f).build())
      actual.map(_.label) shouldBe List("blabel", "alabel")
      actual.map(_.matchCount) shouldBe List(3, 3)
      actual.map(_.probability) shouldEqual List(0.4666667f, 0.43333337f)
      actual match {
        case primary :: secondary :: Nil => {
          primary.significantFeatures should contain theSameElementsAs List(
            (RuleFeature("/str[ij]ng/"),0.6f), (RuleFeature("is"),0.6f))
          secondary.significantFeatures should contain theSameElementsAs List(
            (RuleFeature("/str[ij]ng/"),0.3f), (RuleFeature("is"),0.8f),
            (Word("monkey"), 0.6f))
        }
        case _ => throw new RuntimeException("Expected 2-item list")
      }
    }
  }
}

/**
  * Fake class to make it easy to test combining results.
  * @param labels
  */
case class FakeMCModel(labels: List[String])
    extends PredictModel[Classification]
    with Archivable[FakeMCModel, FakeMCModelLoader] {

  def predict(document: Document,
      options: PredictOptions): Seq[Classification] = {
    labels.map(label => label match {
      case "alabel" if options.includeSignificantFeatures => {
        Classification(label, 0.2f, 1, 0, Seq(Word("monkey") -> 0.6f))
      }
      case _ => {
        Classification(label, 0.2f, 1, 0, Seq())
      }
    })
  }

  override def getFeaturesUsed(): Vector = ???

  override def save(writer: Alloy.Writer): Option[JObject] = {
    val out = writer.resource("stuff")
    Codec.VLuint.write(out, labels.size)
    labels.foreach(label => Codec.String.write(out, label))
    out.close()
    None
  }
}

class FakeMCModelLoader extends ArchiveLoader[FakeMCModel] {
  override def load(engine: Engine, reader: Option[Alloy.Reader], config: Option[JObject]): FakeMCModel = {
    val in = reader.get.resource("stuff")
    val size = Codec.VLuint.read(in)
    val labels = (0 until size).map(_ => {
      val label = Codec.String.read(in)
      label
    })
    in.close()
    new FakeMCModel(labels.toList)
  }
}

/**
  * Helps create a more manageable feature pipeline.
  */
private [this] class MetadataNumberExtractor extends FeatureTransformer with TerminableTransformer {
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
