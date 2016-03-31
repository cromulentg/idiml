package com.idibon.ml.feature.tokenizer

import com.idibon.ml.alloy.Alloy
import com.idibon.ml.common.{Archivable, ArchiveLoader, Engine}
import com.idibon.ml.feature.{Chain, Feature, FeatureTransformer}
import com.idibon.ml.feature.language.LanguageCode
import com.idibon.ml.feature.contenttype.{ContentType, ContentTypeCode}

import com.ibm.icu.util.ULocale
import org.json4s.JObject
import org.json4s.JsonDSL._

/** Tokenization FeatureTransformer */
class TokenTransformer extends FeatureTransformer {

  /** Tokenizes a String, optionally using language-specific rules
    *
    * If more than one input string is provided, the result will be the
    * concatenation of the tokenized results of all input strings.
    *
    * @param content the string to tokenize, represented as a Feature
    * @param language the primary language for the document
    * @param contentType detected document content type
    * @return all of the tokens in content
    */
  def apply(content: Feature[String], language: Feature[LanguageCode],
    contentType: Feature[ContentType]): Seq[Token] = {
    ICUTokenizer.tokenize(content.get, contentType.get.code,
      language.get.icuLocale.getOrElse(ULocale.US))
  }
}

/** Tokenization transform for sequence classifiers
  *
  * Discards tokens that are not included in the accept list, producing
  * a Chain of tokens.
  */
class ChainTokenTransformer(accept: Seq[Tag.Value]) extends FeatureTransformer
    with Archivable[ChainTokenTransformer, ChainTokenTransformerLoader] {

  // generate a bitmask of the accepted tokens, for quick filtering
  private[this] val _tokenMask = accept.map(_.id)
    .foldLeft(0)((accum, id) => accum | (1 << id))

  def apply(content: Feature[String], language: Feature[LanguageCode],
      contentType: Feature[ContentType]): Chain[Token] = {
    Chain(ICUTokenizer.tokenize(content.get, contentType.get.code,
      language.get.icuLocale.getOrElse(ULocale.US))
      .filter(token => (_tokenMask & (1 << token.tag.id)) != 0))
  }

  def save(writer: Alloy.Writer): Option[JObject] = {
    Some(("accept" -> accept.map(_.toString)))
  }
}

/** Paired loader for token chains */
class ChainTokenTransformerLoader extends ArchiveLoader[ChainTokenTransformer] {

  def load(e: Engine, r: Option[Alloy.Reader], config: Option[JObject]) = {
    implicit val formats = org.json4s.DefaultFormats

    val accept = (config.get \ "accept").extract[List[String]]
    new ChainTokenTransformer(accept.map(n => Tag.withName(n)))
  }
}
