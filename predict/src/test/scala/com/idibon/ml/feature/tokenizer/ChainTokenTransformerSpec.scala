package com.idibon.ml.feature.tokenizer

import com.idibon.ml.feature.{Chain, StringFeature}
import com.idibon.ml.feature.language.LanguageCode
import com.idibon.ml.feature.contenttype.{ContentType, ContentTypeCode}
import com.idibon.ml.alloy.{MemoryAlloyReader, MemoryAlloyWriter}
import com.idibon.ml.common.EmbeddedEngine

import org.scalatest.{Matchers, FunSpec}

class ChainTokenTransformerSpec extends FunSpec with Matchers {

  it("should save and load correctly") {
    val archive = collection.mutable.HashMap[String, Array[Byte]]()
    val tok = new ChainTokenTransformer(Seq(Tag.Word, Tag.Punctuation))
    val cfg = tok.save(new MemoryAlloyWriter(archive))
    val tok2 = (new ChainTokenTransformerLoader).load(new EmbeddedEngine,
      Some(new MemoryAlloyReader(archive.toMap)), cfg)
    tok2(StringFeature("Hello, World!"), LanguageCode(Some("eng")),
      ContentType(ContentTypeCode.PlainText)) shouldBe
      Chain(Token("Hello", Tag.Word, 0, 5), Token(",", Tag.Punctuation, 5, 1),
        Token("World", Tag.Word, 7, 5), Token("!", Tag.Punctuation, 12, 1))
  }
}
