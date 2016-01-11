package com.idibon.ml.feature

import org.scalatest.{Matchers, FunSpec}
import java.io._

class FeatureInputOutputStreamSpec extends FunSpec with Matchers {

  it("should gracefully handle more classes than the max cache size") {
    val bos = new ByteArrayOutputStream
    val fos = new FeatureOutputStream(bos, 1)
    val features = List(
      // encoded size, feature
      (7, new tokenizer.Token("foo", tokenizer.Tag.Word, 0, 3)),
      (4, new StringFeature("bar")),
      (6, new tokenizer.Token(":)", tokenizer.Tag.Punctuation, 3, 2))
    )

    for (f <- features) fos.writeFeature(f._2)

    val bytes = bos.toByteArray
    var cursor = 0
    for (f <- features) {
      bytes(cursor) shouldBe 0
      cursor += f._1  // encoded feature
      cursor += f._2.getClass.getName.length + 1  // class name (plus length)
      cursor += 1 // 1-byte cache index
    }

    val fis = new FeatureInputStream(new ByteArrayInputStream(bytes))
    for (f <- features) fis.readFeature shouldBe f._2
  }

  it("should cache class types") {
    val bos = new ByteArrayOutputStream
    val fos = new FeatureOutputStream(bos)
    val features = List(
      new tokenizer.Token("foo", tokenizer.Tag.Word, 0, 3),
      new tokenizer.Token("bar", tokenizer.Tag.Word, 3, 3),
      new StringFeature("hello"),
      new StringFeature("world"),
      new tokenizer.Token("foo", tokenizer.Tag.Word, 0, 3)
    )

    for (f <- features) fos.writeFeature(f)

    val bytes = bos.toByteArray
    // the first byte of each feature should be an "insert at index 0" command
    bytes(0) shouldBe 0
    bytes("com.idibon.ml.feature.tokenizer.Token".length + 9) shouldBe -128
    bytes("com.idibon.ml.feature.tokenizer.Token".length + 17) shouldBe 1
    bytes("com.idibon.ml.feature.tokenizer.Token".length +
      "com.idibon.ml.feature.StringFeature".length + 25) shouldBe -127
    bytes("com.idibon.ml.feature.tokenizer.Token".length +
      "com.idibon.ml.feature.StringFeature".length + 32) shouldBe -128

    val fis = new FeatureInputStream(new ByteArrayInputStream(bytes))
    for (f <- features) fis.readFeature shouldBe f
  }
}

