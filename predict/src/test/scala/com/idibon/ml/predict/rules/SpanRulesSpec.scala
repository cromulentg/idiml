package com.idibon.ml.predict.rules

import java.util.regex.Pattern

import com.idibon.ml.alloy.{MemoryAlloyReader, MemoryAlloyWriter}
import com.idibon.ml.common.EmbeddedEngine
import com.idibon.ml.predict._
import org.apache.spark.mllib.linalg.SparseVector
import org.json4s._
import org.scalatest.{BeforeAndAfter, FunSpec, Matchers}

import scala.collection.mutable.HashMap


/**
  * Class to test the Span Rules Model.
  */
class SpanRulesSpec extends FunSpec with Matchers with BeforeAndAfter {

  describe("getRuleMatches") {

    before {
    }

    it("should work on no rules") {
      val spanRules = new SpanRules("a-label", "a-label", List())
      val results = spanRules.getRuleMatches("this is a string")
      results.size shouldBe 0
      results shouldBe Map()
    }

    it("should work on an exact rule") {
      val spanRules = new SpanRules("a-label", "a-label", List(("is", 1.0f)))
      val results = spanRules.getRuleMatches("this is a string")
      results.size shouldBe 1
      results.get("is") shouldBe Some(Seq((2,4), (5,7))) // 2: thIS and IS
    }

    it("should work on a regex rule") {
      val spanRules = new SpanRules("a-label", "a-label", List(("/str[ij]ng/", 1.0f)))
      val results = spanRules.getRuleMatches("this is a string")
      results.size shouldBe 1
      results.get("/str[ij]ng/") shouldBe Some(Seq((10,16)))
    }

    it("should work on two rules") {
      val spanRules = new SpanRules("a-label", "a-label", List(("/str[ij]ng/", 1.0f), ("is", 1.0f)))
      val results = spanRules.getRuleMatches("this is a string with is")
      results.size shouldBe 2
      results.get("/str[ij]ng/") shouldBe Some(Seq((10,16)))
      results.get("is") shouldBe Some(Seq((2,4), (5,7), (22,24)))
    }

    it("should work as expected on badly formed regex rules") {
      //missing front / so should be literal & bad regex
      val spanRules = new SpanRules("a-label", "a-label", List(("str[ij]]ng/", 1.0f), ("/*._<?>asdis/", 1.0f)))
      val results = spanRules.getRuleMatches("this is a string with is")
      results.size shouldBe 0 //no matches
      val results2 = spanRules.getRuleMatches("/*._<?>asdis/")
      results2.size shouldBe 0 //no matches
      val results3 = spanRules.getRuleMatches("str[ij]]ng/")
      results3.size shouldBe 1 //one match
    }

    it("should work in the case a rule is not in the cache") {
      val spanRules = new SpanRules("a-label", "a-label", List(("/str[ij]ng/", 1.0f), ("/is\\u{1FFFF}/", 1.0f)))
      val results = spanRules.getRuleMatches("this is a string with is")
      results.size shouldBe 1
      results.get("/str[ij]ng/") shouldBe Some(Seq((10,16)))
      results.get("is") shouldBe None
    }
  }

  describe("get matches tests") {
    val spanRules = new SpanRules("", "", List())
    it("should handle named group regular expressions") {
      val rules = RulesHelpers.compileRules(Seq("/(?i)(m(rs?|s)|dr)\\.?\\s*(?<LastName>[a-z]+)/"))
      val pattern = rules.head._2.get
      val matcher = pattern.matcher("Dr. Jones, Jones, calling Dr. Jones!")
      val actual = spanRules.getMatches(matcher, "LastName")
      actual shouldBe List((4,9), (30,35))
    }
    it("should handle no named group in the regular expression") {
      val rules = RulesHelpers.compileRules(Seq("/(?i)(m(rs?|s)|dr)\\.?\\s*([a-z]+)/"))
      val pattern = rules.head._2.get
      val matcher = pattern.matcher("Dr. Jones, Jones, calling Dr. Jones!")
      val actual = spanRules.getMatches(matcher, "LastName")
      actual shouldBe List((0,9), (26,35))
    }
    it("should handle no matches") {
      val rules = RulesHelpers.compileRules(Seq("/(?i)(m(rs?|s)|dr)\\.?\\s*(?<LastName>[a-z]+)/"))
      val pattern = rules.head._2.get
      val matcher = pattern.matcher("asdfasdf asdfdasfads fadsf a")
      val actual = spanRules.getMatches(matcher, "LastName")
      actual shouldBe List()
    }
  }

  describe("get matcher group index tests") {
    val spanRules = new SpanRules("", "", List())
    it("should handle no named group") {
      val rules = RulesHelpers.compileRules(Seq("/(?i)(m(rs?|s)|dr)\\.?\\s*([a-z]+)/"))
      val pattern = rules.head._2.get
      val matcher = pattern.matcher("Dr. Jones!")
      matcher.find() shouldBe true
      val actual = spanRules.getMatcherGroupIndex(matcher, "LastName")
      actual shouldBe 0
    }
    it("should handle a named group but not the one we want") {
      val rules = RulesHelpers.compileRules(Seq("/(?i)(m(rs?|s)|dr)\\.?\\s*(?<Fake>[a-z]+)/"))
      val pattern = rules.head._2.get
      val matcher = pattern.matcher("Dr. Jones!")
      matcher.find() shouldBe true
      val actual = spanRules.getMatcherGroupIndex(matcher, "LastName")
      actual shouldBe 0
    }
    it("should handle a named group") {
      val rules = RulesHelpers.compileRules(Seq("/(?i)(m(rs?|s)|dr)\\.?\\s*(?<LastName>[a-z]+)/"))
      val pattern = rules.head._2.get
      val matcher = pattern.matcher("Dr. Jones!")
      matcher.find() shouldBe true
      val actual = spanRules.getMatcherGroupIndex(matcher, "LastName")
      actual shouldBe 3
    }
  }

  describe("constructor") {
    it("Should not filter out good rule weights") {
      val spanRules = new SpanRules("b-label", "b-label", List(("/str[ij]ng/", 1.0f), ("is", 0.0f), ("mon", 0.5f)))
      spanRules.rulesCache.size shouldBe 3
    }

    it("Should filter out bad rule weights") {
      val spanRules = new SpanRules("b-label", "b-label", List(("/str[ij]ng/", 1.5f), ("is", -0.5f)))
      spanRules.rulesCache.size shouldBe 0
    }
  }

  describe("spanPredict") {
    it("creates empty result seq when there are no matches") {
      val spanRules = new SpanRules("a-label", "a-label", List())
      val results = spanRules.spanPredict("this is a string")
      results.size shouldBe 0
      results shouldBe Seq()
    }
    it("creates spans with whitelisted flag") {
      val spanRules = new SpanRules("a-label", "a-label", List(("is", 1.0f)))
      val results = spanRules.spanPredict("this is a string")
      results.size shouldBe 2
      results(0) shouldBe Span("a-label", 1.0f, 3, 2, 2)
      results(1) shouldBe Span("a-label", 1.0f, 3, 5, 2)
    }
    it("creates spans with blacklisted flag") {
      val spanRules = new SpanRules("a-label", "a-label", List(("is", 0.0f)))
      val results = spanRules.spanPredict("this is a string")
      results.size shouldBe 2
      results(0) shouldBe Span("a-label", 0.0f, 3, 2, 2)
      results(1) shouldBe Span("a-label", 0.0f, 3, 5, 2)
    }
    it("creates normal spans on a match") {
      val spanRules = new SpanRules("a-label", "a-label", List(("is", 0.5f)))
      val results = spanRules.spanPredict("this is a string")
      results.size shouldBe 2
      results(0) shouldBe Span("a-label", 0.5f, 2, 2, 2)
      results(1) shouldBe Span("a-label", 0.5f, 2, 5, 2)
    }
  }

  describe("predict from JSON object") {
    it("Should predict properly when there is a match") {
      val doc = new JObject(List("content" -> new JString("this is some content")))
      val model = new SpanRules("a-label", "a-label", List(("/str[ij]ng/", 1.0f), ("is", 1.0f)))
      val actual = model.predict(Document.document(doc), PredictOptions.DEFAULT)
      actual.size shouldBe 2
      actual(0).label shouldBe "a-label"
      actual(0).probability shouldBe 1.0f
      actual(0).offset shouldBe 2
      actual(1).label shouldBe "a-label"
      actual(1).probability shouldBe 1.0f
      actual(1).offset shouldBe 5
    }

    it("Should predict properly when there is no match") {
      val doc = new JObject(List("content" -> new JString("this is some content")))
      val model = new SpanRules("a-label", "a-label", List(("/str[ij]ng/", 1.0f), ("/no[thing].*/", 1.0f)))
      val actual = model.predict(Document.document(doc), PredictOptions.DEFAULT)
      actual.size shouldBe 0
    }
  }

    describe("save and load") {
      it("should save an empty map and load it") {
        val archive = HashMap[String, Array[Byte]]()
        val spanRules = new SpanRules("b-label", "b-label", List())
        val jsonConfig = spanRules.save(new MemoryAlloyWriter(archive))
        jsonConfig shouldBe Some(JObject(
          List(
            ("labelUUID", JString("b-label")),
            ("labelHuman", JString("b-label"))
          )))
        val spanRulesLoad = (new SpanRulesLoader).load(
          new EmbeddedEngine, Some(new MemoryAlloyReader(archive.toMap)), jsonConfig)
        spanRulesLoad.rules shouldBe spanRules.rules
        spanRulesLoad.labelHuman shouldBe spanRules.labelHuman
        spanRulesLoad.labelUUID shouldBe spanRules.labelUUID
      }

      it("should save a valid map") {
        val archive = HashMap[String, Array[Byte]]()
        val spanRules = new SpanRules("b-label", "b-label", List(("/str[ij]ng/", 0.3f), ("is", 0.10f)))
        val jsonConfig = spanRules.save(new MemoryAlloyWriter(archive))
        jsonConfig shouldBe Some(JObject(
          List(
            ("labelUUID", JString("b-label")),
            ("labelHuman", JString("b-label"))
          )))
        val spanRulesLoad = (new SpanRulesLoader).load(
          new EmbeddedEngine, Some(new MemoryAlloyReader(archive.toMap)), jsonConfig)
        spanRulesLoad.rules shouldBe spanRules.rules
        spanRulesLoad.labelHuman shouldBe spanRules.labelHuman
        spanRulesLoad.labelUUID shouldBe spanRules.labelUUID
      }
    }

  describe("exercised features used") {
    it("returns a sparse vector") {
      val model = new SpanRules("a-label", "a-label", List(("/str[ij]ng/", 1.0f), ("/no[thing].*/", 1.0f)))
      model.getFeaturesUsed() shouldBe new SparseVector(1, Array(0), Array(0))
    }
  }

}
