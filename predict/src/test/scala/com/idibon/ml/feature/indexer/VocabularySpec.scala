package com.idibon.ml.feature.indexer

import java.io.{ByteArrayInputStream, ByteArrayOutputStream}

import com.idibon.ml.feature.{Feature, StringFeature,
  FeatureInputStream, FeatureOutputStream}
import com.idibon.ml.test.VerifyLogging

import org.scalatest.{Matchers, FunSpec, BeforeAndAfterAll}

class VocabularySpec extends FunSpec with Matchers
    with VerifyLogging with BeforeAndAfterAll {

  override val loggerName = Vocabulary.getClass.getName

  override def beforeAll = {
    super.beforeAll
    initializeLogging
  }

  override def afterAll = {
    shutdownLogging
    super.afterAll
  }

  describe("equals") {

    it("allows frozen mutable vocabularies to equal immutable vocabularies") {
      val mutable = new MutableVocabulary
      mutable(StringFeature("foo")) shouldBe 0
      mutable(StringFeature("bar")) shouldBe 1
      val frozen = mutable.freeze

      val os = new ByteArrayOutputStream
      frozen.save(new FeatureOutputStream(os))
      val fis = new FeatureInputStream(new ByteArrayInputStream(os.toByteArray))
      val immutable = Vocabulary.load(fis)

      immutable shouldBe an [ImmutableVocabulary]
      immutable shouldBe frozen
      immutable should not be mutable
      frozen should not be mutable
    }

    it("warns if unsaveable features exist in the domain") {
      val mutable = new MutableVocabulary
      mutable(StringFeature("foo")) shouldBe 0
      mutable(UnsaveableFeature("bar")) shouldBe 1
      val os = new ByteArrayOutputStream
      mutable.save(new FeatureOutputStream(os))
      loggedMessages should include regex "Vocabulary has 1 unsaveable dimensions"
      val fis = new FeatureInputStream(new ByteArrayInputStream(os.toByteArray))
      val loaded = Vocabulary.load(fis)
      loaded shouldBe a [MutableVocabulary]
      loaded should not be mutable
      loaded.size shouldBe 2
      loaded.assigned shouldBe 1
      loaded.invert(0) shouldBe Some(StringFeature("foo"))
      loaded.invert(1) shouldBe None
    }
  }
}

case class UnsaveableFeature(s: String) extends Feature[String] {
  def get = s
  def getHumanReadableString = Some(s)
}
