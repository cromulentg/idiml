package com.idibon.ml.predict.rules

import java.util.regex.Pattern
import scala.collection.mutable.HashMap

import com.idibon.ml.alloy.{MemoryAlloyReader, MemoryAlloyWriter}
import com.idibon.ml.predict._
import com.idibon.ml.common.EmbeddedEngine
import org.apache.spark.mllib.linalg.SparseVector
import org.json4s._
import org.scalatest.{Matchers, BeforeAndAfter, FunSpec}


/**
  * Class to test the Document Rules Model.
  */
class DocumentRulesSpec extends FunSpec with Matchers with BeforeAndAfter {

  describe("getDocumentMatchCounts") {

    before {
    }

    it("should work on no rules") {
      val docRules = new DocumentRules("a-label", List())
      val results = docRules.getDocumentMatchCounts("this is a string")
      results.size shouldBe 0
    }

    it("should work on an exact rule") {
      val docRules = new DocumentRules("a-label", List(("is", 1.0f)))
      val results = docRules.getDocumentMatchCounts("this is a string")
      results.size shouldBe 1
      results.get("is") shouldBe Some(2) // 2: thIS and IS
    }

    it("should work on a regex rule") {
      val docRules = new DocumentRules("a-label", List(("/str[ij]ng/", 1.0f)))
      val results = docRules.getDocumentMatchCounts("this is a string")
      results.size shouldBe 1
      results.get("/str[ij]ng/") shouldBe Some(1)
    }

    it("should work on two rules") {
      val docRules = new DocumentRules("a-label", List(("/str[ij]ng/", 1.0f), ("is", 1.0f)))
      val results = docRules.getDocumentMatchCounts("this is a string with is")
      results.size shouldBe 2
      results.get("/str[ij]ng/") shouldBe Some(1)
      results.get("is") shouldBe Some(3)
    }

    it("should work as expected on badly formed regex rules") {
      //missing front / so should be literal & bad regex
      val docRules = new DocumentRules("a-label", List(("str[ij]]ng/", 1.0f), ("/*._<?>asdis/", 1.0f)))
      val results = docRules.getDocumentMatchCounts("this is a string with is")
      results.size shouldBe 0 //no matches
      val results2 = docRules.getDocumentMatchCounts("/*._<?>asdis/")
      results2.size shouldBe 0 //no matches
      val results3 = docRules.getDocumentMatchCounts("str[ij]]ng/")
      results3.size shouldBe 1 //one matche
    }

    it("should work in the case a rule is not in the cache") {
      val docRules = new DocumentRules("a-label", List(("/str[ij]ng/", 1.0f), ("/is\\u{1FFFF}/", 1.0f)))
      val results = docRules.getDocumentMatchCounts("this is a string with is")
      results.size shouldBe 1
      results.get("/str[ij]ng/") shouldBe Some(1)
      results.get("is") shouldBe None
    }
  }

  describe("save and load") {
    it("should save an empty map and load it") {
      val archive = HashMap[String, Array[Byte]]()
      val docRules = new DocumentRules("b-label", List())
      val jsonConfig = docRules.save(new MemoryAlloyWriter(archive))
      jsonConfig shouldBe Some(JObject(List(("label", JString("b-label")))))
      //bogus stuff that should be overwritten
      val docRulesLoad = (new DocumentRulesLoader).load(
        new EmbeddedEngine, Some(new MemoryAlloyReader(archive.toMap)), jsonConfig)
      docRulesLoad.rules shouldBe docRules.rules
      docRulesLoad.label shouldBe docRules.label
    }

    it("should save a valid map") {
      val archive = HashMap[String, Array[Byte]]()
      val docRules = new DocumentRules("b-label", List(("/str[ij]ng/", 0.3f), ("is", 0.10f)))
      val jsonConfig = docRules.save(new MemoryAlloyWriter(archive))
      jsonConfig shouldBe Some(JObject(List(("label", JString("b-label")))))
      //bogus stuff that should be overwritten
      val docRulesLoad = (new DocumentRulesLoader).load(
        new EmbeddedEngine, Some(new MemoryAlloyReader(archive.toMap)), jsonConfig)
      docRulesLoad.rules shouldBe docRules.rules
      docRulesLoad.label shouldBe docRules.label
    }
  }

  describe("getMatches") {
    it("should return empty list on no matches") {
      val pat = Pattern.compile("is", Pattern.LITERAL | Pattern.CASE_INSENSITIVE)
      val matcher = pat.matcher("no matches whatsoever")
      (new DocumentRules("a-label", List())).getMatches(matcher) shouldBe List[(Int, Int)]()
    }

    it("should return a valid list with a valid match") {
      val pat = Pattern.compile("is", Pattern.LITERAL | Pattern.CASE_INSENSITIVE)
      val matcher = pat.matcher("is is is")
      new DocumentRules("a-label", List()).getMatches(matcher) shouldBe List[(Int, Int)]((0, 2), (3, 5), (6, 8))
    }
  }

  describe("constructor") {
    it("Should not filter out good rule weights") {
      val docRules = new DocumentRules("b-label", List(("/str[ij]ng/", 1.0f), ("is", 0.0f), ("mon", 0.5f)))
      docRules.rulesCache.size shouldBe 3
    }

    it("Should filter out bad rule weights") {
      val docRules = new DocumentRules("b-label", List(("/str[ij]ng/", 1.5f), ("is", -0.5f)))
      docRules.rulesCache.size shouldBe 0
    }
  }

  describe("docPredict") {
    it("Should return significant features") {
      val docRules = new DocumentRules("b-label", List(("/str[ij]ng/", 0.5f), ("is", 0.5f)))
      val actual = docRules.docPredict("string matching is working", true)
      actual.label shouldBe "b-label"
      actual.matchCount shouldBe 2
      actual.probability shouldEqual 0.5f
      actual.significantFeatures should contain theSameElementsAs List(
        (RuleFeature("/str[ij]ng/"), 0.5f), (RuleFeature("is"), 0.5f))
    }

    it("Should return whitelist significant feature when whitelist overrides") {
      val docRules = new DocumentRules("b-label", List(("/str[ij]ng/", 1.0f), ("is", 0.5f)))
      val actual = docRules.docPredict("string matching is working", true)
      actual.label shouldBe "b-label"
      actual.matchCount shouldBe 1
      actual.probability shouldEqual 1.0f
      actual.significantFeatures shouldEqual List((RuleFeature("/str[ij]ng/"), 1.0f))
    }

    it("Should return whitelist significant feature when blacklist overrides") {
      val docRules = new DocumentRules("b-label", List(("/str[ij]ng/", 0.0f), ("is", 0.5f)))
      val actual = docRules.docPredict("string strjng matching is working", true)
      actual.label shouldBe "b-label"
      actual.matchCount shouldBe 2
      actual.probability shouldEqual 0.0f
      actual.significantFeatures shouldEqual List((RuleFeature("/str[ij]ng/"), 0.0f))
    }

    it("Should return a document prediction result") {
      val docRules = new DocumentRules("b-label", List(("/str[ij]ng/", 0.5f), ("is", 0.5f)))
      val actual = docRules.docPredict("string matching is working", false)
      actual.label shouldBe "b-label"
      actual.matchCount shouldBe 2
      actual.probability shouldEqual 0.5f
      actual.significantFeatures shouldEqual List()
    }
  }

  describe("isRegexRule") {
    it("correctly identifies regular expression") {
      DocumentRules.isRegexRule("/str[ij]ng/") shouldBe true
    }

    it("correctly identifies non-regular expression") {
      DocumentRules.isRegexRule("/str[ij]ng") shouldBe false
      DocumentRules.isRegexRule("str[ij]ng") shouldBe false
      DocumentRules.isRegexRule(null) shouldBe false
    }
  }

  describe("calculatePseudoProbability") {
    it("should skip 0 values in count map") {
      val docRules = new DocumentRules("b-label", List(("/str[ij]ng/", 0.5f), ("is", 0.5f)))
      val countMap = Map[String, Int](("/str[ij]ng/", 0), ("is", 0))
      val (prob, count, worb) = docRules.calculatePseudoProbability(countMap)
      prob shouldBe 0.0
      count shouldBe 0.0
      worb shouldBe false
    }

    it("should return correctly with no whitelist or blacklist hit when a normal rule is hit") {
      val docRules = new DocumentRules("b-label", List(("/str[ij]ng/", 0.5f), ("is", 0.5f)))
      val countMap = Map[String, Int](("/str[ij]ng/", 1), ("is", 1))
      val (prob, count, worb) = docRules.calculatePseudoProbability(countMap)
      prob shouldBe 0.5f
      count shouldBe 2
      worb shouldBe false
    }

    it("should return whitelist correctly") {
      val docRules = new DocumentRules("b-label", List(("/str[ij]ng/", 1.0f), ("is", 0.5f)))
      val countMap = Map[String, Int](("/str[ij]ng/", 1))
      val (prob, count, worb) = docRules.calculatePseudoProbability(countMap)
      prob shouldBe 1.0f
      count shouldBe 1
      worb shouldBe true
    }

    it("should return blacklist correctly") {
      val docRules = new DocumentRules("b-label", List(("/str[ij]ng/", 0.0f), ("is", 0.5f)))
      val countMap = Map[String, Int](("/str[ij]ng/", 1))
      val (prob, count, worb) = docRules.calculatePseudoProbability(countMap)
      prob shouldBe 0.0
      count shouldBe 1
      worb shouldBe true
    }

    it("should merge blacklist and whitelist correctly if both present") {
      val docRules = new DocumentRules("b-label", List(("/str[ij]ng/", 0.0f), ("is", 1.0f)))
      val countMap = Map[String, Int](("/str[ij]ng/", 2), ("is", 8))
      val (prob, count, worb) = docRules.calculatePseudoProbability(countMap)
      prob shouldBe 0.8f
      count shouldBe 10
      worb shouldBe true
    }

    it("should handle overriding rule hits if blacklist matched") {
      val docRules = new DocumentRules("b-label", List(("/str[ij]ng/", 0.0f), ("is", 0.5f)))
      val countMap = Map[String, Int](("/str[ij]ng/", 2), ("is", 8))
      val (prob, count, worb) = docRules.calculatePseudoProbability(countMap)
      prob shouldBe 0.0f
      count shouldBe 2
      worb shouldBe true
    }

    it("should handle overriding rule hits if whitelist matched") {
      val docRules = new DocumentRules("b-label", List(("/str[ij]ng/", 1.0f), ("is", 0.5f)))
      val countMap = Map[String, Int](("/str[ij]ng/", 2), ("is", 8))
      val (prob, count, worb) = docRules.calculatePseudoProbability(countMap)
      prob shouldBe 1.0f
      count shouldBe 2
      worb shouldBe true
    }

    it("should return correctly when map passed is empty") {
      val docRules = new DocumentRules("b-label", List(("/str[ij]ng/", 1.0f), ("is", 0.5f)))
      val countMap = Map[String, Int]()
      val (prob, count, worb) = docRules.calculatePseudoProbability(countMap)
      prob shouldBe 0.0f
      count shouldBe 0
      worb shouldBe false
    }
  }

  describe("predict from JSON object") {
    it("Should predict properly when there is a match") {
      val doc = new JObject(List("content" -> new JString("this is some content")))
      val model = new DocumentRules("a-label", List(("/str[ij]ng/", 1.0f), ("is", 1.0f)))
      val actual = model.predict(Document.document(doc), PredictOptions.DEFAULT).head
      actual.label shouldBe "a-label"
      actual.probability shouldBe 1.0f
    }

    it("Should predict properly when there is no match") {
      val doc = new JObject(List("content" -> new JString("this is some content")))
      val model = new DocumentRules("a-label", List(("/str[ij]ng/", 1.0f), ("/no[thing].*/", 1.0f)))
      val actual = model.predict(Document.document(doc), PredictOptions.DEFAULT).head
      actual.label shouldBe "a-label"
      actual.probability shouldBe 0.0f
    }
  }

  describe("exercised features used") {
    it("returns a sparse vector") {
      val model = new DocumentRules("a-label", List(("/str[ij]ng/", 1.0f), ("/no[thing].*/", 1.0f)))
      model.getFeaturesUsed() shouldBe new SparseVector(1, Array(0), Array(0))
    }
  }

  describe("get") {
    it("RuleFeature.get should output the rule string") {
      val ruleFeatureHello = new RuleFeature("Hello")
      val ruleFeatureWorld = new RuleFeature("World")

      ruleFeatureHello.get shouldBe "Hello"
      ruleFeatureWorld.get shouldBe "World"
    }

    it("RuleFeature.getHumanReadableString should output human-readable strings") {
      val ruleFeatureHello = new RuleFeature("Hello")
      val ruleFeatureWorld = new RuleFeature("World")

      ruleFeatureHello.getHumanReadableString shouldBe Some("Hello")
      ruleFeatureWorld.getHumanReadableString shouldBe Some("World")
    }
  }

}
