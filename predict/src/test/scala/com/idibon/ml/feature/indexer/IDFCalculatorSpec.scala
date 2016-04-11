package com.idibon.ml.feature.indexer

import java.io.{ByteArrayInputStream, ByteArrayOutputStream}

import com.idibon.ml.feature.{FeatureInputStream, FeatureOutputStream}

import org.scalatest.{Matchers, FunSpec}

import scala.collection.mutable.ArrayBuffer

class IDFCalculatorSpec extends FunSpec with Matchers {

  describe("computes IDF values tests") {
    it("handles empty numbers") {
      IDFCalculator.computeIDFs(0, IndexedSeq(), 1) shouldBe Map()
    }
    it("handles filtering out all counts") {
      IDFCalculator.computeIDFs(10, IndexedSeq(1, 1, 1), 2) shouldBe Map()
    }
    it("handles 0 min doc count") {
      val actual = IDFCalculator.computeIDFs(10, IndexedSeq(1, 1, 0), 0)
      actual.size shouldBe 2
      actual(0) shouldBe 2.302585092994046 +- 0.000005
      actual(1) shouldBe 2.302585092994046 +- 0.000005
    }
    it("handles computing correct values") {
      val actual = IDFCalculator.computeIDFs(10, IndexedSeq(1, 1, 1), 1)
      actual.size shouldBe 3
      actual(0) shouldBe 2.302585092994046 +- 0.000005
      actual(1) shouldBe 2.302585092994046 +- 0.000005
      actual(2) shouldBe 2.302585092994046 +- 0.000005
    }
  }

  describe("save & load tests") {
    it("saves & loads immutable idf calc") {
      val idf = new ImmutableIDFCalculator(Map(0 -> 0.4, 3 -> 0.3))
      val os = new ByteArrayOutputStream
      idf.save(new FeatureOutputStream(os))
      val fis = new FeatureInputStream(new ByteArrayInputStream(os.toByteArray))
      val immutable = IDFCalculator.load(fis)
      immutable shouldBe idf
    }
    it("saves & loads mutable idf calc") {
      val counter = new ArrayBuffer[Int]()
      counter ++= Seq(1, 5, 2, 0, 2, 6)
      val idf = new MutableIDFCalculator(counter, 10)
      val os = new ByteArrayOutputStream
      idf.save(new FeatureOutputStream(os))
      val fis = new FeatureInputStream(new ByteArrayInputStream(os.toByteArray))
      val mutable = IDFCalculator.load(fis)
      mutable shouldBe idf
    }

    it("saves frozen mutable & loads immutable idf calc") {
      val counts = Seq(1, 5, 2, 0, 2, 6)
      val counter = new ArrayBuffer[Int]()
      counter ++= Seq(1, 5, 2, 0, 2, 6)
      val idf = new MutableIDFCalculator(counter, 10)
      idf.frozen = true
      val os = new ByteArrayOutputStream
      idf.save(new FeatureOutputStream(os))
      val fis = new FeatureInputStream(new ByteArrayInputStream(os.toByteArray))
      val immutable = IDFCalculator.load(fis)
      immutable.isInstanceOf[ImmutableIDFCalculator] shouldBe true
      val numerator = Math.log(10)
      immutable.inverseDocumentFrequency(0) shouldBe (numerator - Math.log(counts(0)))
      immutable.inverseDocumentFrequency(1) shouldBe (numerator - Math.log(counts(1)))
      immutable.inverseDocumentFrequency(2) shouldBe (numerator - Math.log(counts(2)))
      immutable.inverseDocumentFrequency(3) shouldBe 0.0
      immutable.inverseDocumentFrequency(4) shouldBe (numerator - Math.log(counts(4)))
      immutable.inverseDocumentFrequency(5) shouldBe (numerator - Math.log(counts(5)))
    }
  }
}

class ImmutableIDFCalculatorSpec extends FunSpec with Matchers {
  val idf = new ImmutableIDFCalculator(Map(0 -> 0.4, 3 -> 0.3))
  describe("inverse doc frequency tests") {
    it("get correct value") {
      idf.inverseDocumentFrequency(Seq(0, 3)) shouldBe Seq(0.4, 0.3)
    }
    it("handles oov") {
      idf.inverseDocumentFrequency(Seq(1, 2, 4)) shouldBe Seq(0.0, 0.0, 0.0)
    }
  }
  describe("noop method tests") {
    it("should not change any state or do anything unexpected") {
      idf.incrementTotalDocCount()
      idf.incrementSeenCount(Array(0,1,2,3,4,5))
      idf.freeze
      idf.prune((i: Int) => true)
      idf.minimumDocumentObservations = 3
      idf.minimumDocumentObservations shouldBe 0
    }
  }
}
class MutableIDFCalculatorSpec extends FunSpec with Matchers {

  describe("count related tests") {
    var counter = new ArrayBuffer[Int]()
    counter ++= Seq(1, 5, 2, 0, 2, 6)

    it("increments doc count") {
      val idf = new MutableIDFCalculator(counter, 10)
      idf.incrementTotalDocCount()
      idf.numDocs shouldBe 11
    }
    it("doesn't increment doc count once frozen") {
      val idf = new MutableIDFCalculator(counter, 10)
      idf.incrementTotalDocCount()
      idf.numDocs shouldBe 11
      idf.frozen = true
      idf.incrementTotalDocCount()
      idf.numDocs shouldBe 11
    }
    it("increments feature seen counts") {
      var localCounter = new ArrayBuffer[Int]()
      localCounter ++= Seq(1, 5, 2, 0, 2, 6)
      val idf = new MutableIDFCalculator(localCounter, 10)
      idf.incrementSeenCount(Array(1, 3, 6, 7, 8))
      idf.counter.toArray shouldBe Array(1, 6, 2, 1, 2, 6, 1, 1, 1)
    }
    it("throws assert error when incrementing badly") {
      val idf = new MutableIDFCalculator(counter, 10)
      intercept[AssertionError] {
        idf.incrementSeenCount(Array(11))
      }
    }
    it("doesn't increment seen counts once frozen") {
      var localCounter = new ArrayBuffer[Int]()
      localCounter ++= Seq(1, 5, 2, 0, 2, 6)
      val idf = new MutableIDFCalculator(localCounter, 10)
      idf.incrementSeenCount(Array(1, 3, 6, 7, 8))
      idf.counter.toArray shouldBe Array(1, 6, 2, 1, 2, 6, 1, 1, 1)
      idf.frozen = true
      idf.incrementSeenCount(Array(1, 3, 6, 7, 8))
      idf.counter.toArray shouldBe Array(1, 6, 2, 1, 2, 6, 1, 1, 1)
    }
  }

  describe("inverse doc freq computation related tests") {
    val counter = new ArrayBuffer[Int]()
    counter ++= Seq(1, 5, 2, 0, 2, 6)
    val idf = new MutableIDFCalculator(counter, 10)
    it("computes IDF values") {
      val expected = List(2.302585092994046, 0.0, 1.6094379124341005, 0.5108256237659909)
      val actual = idf.inverseDocumentFrequency(Seq(0, 3, 4, 5))
      actual.zip(expected).foreach({case (a, e) => a shouldBe e +- 0.00005})
    }
    it("handles OOV dimensions") {
      idf.inverseDocumentFrequency(Seq(19, 11, 14, 15)) shouldBe Seq(0.0, 0.0, 0.0, 0.0)
    }
  }

  describe("freeze tests") {
    it("creates correct immutable structure") {
      val counter = new ArrayBuffer[Int]()
      counter ++= Seq(1, 5, 2, 0, 2, 6)
      val idf = new MutableIDFCalculator(counter, 10)
      val actual = idf.freeze
      val expected = new MutableIDFCalculator(counter, 10)
      expected.frozen = true
      actual shouldBe expected
    }
  }

  describe("prune tests") {
    it("resets counts to 0") {
      val counter = new ArrayBuffer[Int]()
      counter ++= Seq(1, 5, 2, 0, 2, 6)
      val idf = new MutableIDFCalculator(counter, 10)
      idf.prune((i: Int) => {
        i % 2 == 0
      })
      idf.counter.toArray shouldBe Array(0, 5, 0, 0, 0, 6)
    }
  }

}

