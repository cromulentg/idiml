package com.idibon.ml.feature.bagofwords

import com.idibon.ml.feature.{Feature, FeatureTransformer}
import com.idibon.ml.feature.tokenizer.{Token, Tag}
import com.idibon.ml.feature.language.LanguageCode
import com.idibon.ml.common.{Archivable, ArchiveLoader, Engine}
import com.idibon.ml.alloy.Alloy
import org.json4s._
import org.json4s.JsonDSL._

import com.ibm.icu.lang.UCharacter

object CaseTransform extends Enumeration {
  // TODO: add more case fold choices here
  val ToLower,  /* convert all words to lower case */
    ToUpper,    /* convert all words to upper case */
    None        /* no change in case */ = Value
}

/** BagOfWords FeatureTransformer
  *
  * Produces a bag of words. i.e. a Word Feature for each token, optionally
  * applying language-specific case folding and filtering by the token type
  *
  * @param accept - a list of accepted (i.e., converted) token types
  * @param transform - a case folding operation to apply
  */
class BagOfWordsTransformer(accept: Seq[Tag.Value],
  transform: CaseTransform.Value) extends FeatureTransformer
    with Archivable[BagOfWordsTransformer, BagOfWordsTransformerLoader] {

  // for performance, convert the list of accepted token tags into a bitmask
  private[this] val _tokenMask = accept.map(_.id)
    .foldLeft(0)((accum, id) => accum | (1 << id))

  /** Produces a bag of words from a sequence of Tokens
    *
    * @param tokens - the generated tokens
    * @param language - the primary language identified for the token stream
    * @return bag of words represented as a sequence of Word features.
    */
  def apply(tokens: Seq[Feature[Token]], language: Feature[LanguageCode]):
      Seq[Word] = {
    val accepted = tokens.filter(t => (_tokenMask & (1 << t.get.tag.id)) != 0)
    this.transform match {
      case CaseTransform.ToLower => {
        language.get.icuLocale.map(locale => {
          accepted.map(t => Word(UCharacter.toLowerCase(locale, t.get.content)))
        }).getOrElse({
          accepted.map(t => Word(UCharacter.toLowerCase(t.get.content)))
        })
      }
      case CaseTransform.ToUpper => {
        language.get.icuLocale.map(locale => {
          accepted.map(t => Word(UCharacter.toUpperCase(locale, t.get.content)))
        }).getOrElse({
          accepted.map(t => Word(UCharacter.toUpperCase(t.get.content)))
        })
      }
      case CaseTransform.None => {
        accepted.map(t => Word(t.get.content))
      }
    }
  }

  /** Saves the configuration data for this transform to an Alloy */
  def save(writer: Alloy.Writer): Option[JObject] = {
    Some(("accept" -> this.accept.map(_.toString)) ~
      ("transform" -> this.transform.toString))
  }
}

/** Paired loader for BagOfWordsTransformer */
class BagOfWordsTransformerLoader extends ArchiveLoader[BagOfWordsTransformer] {
  def load(engine: Engine, reader: Alloy.Reader, config: Option[JObject]) = {
    implicit val formats = DefaultFormats

    val bowConfig = config.get.extract[BagOfWordsConfig]
    val accept = bowConfig.accept.map(t => Tag.withName(t))
    val transform = CaseTransform.withName(bowConfig.transform)
    new BagOfWordsTransformer(accept, transform)
  }
}

/** JSON configuration data for the bag-of-words transform
  *
  * @param accept - a list of names of accepted token types (must be Tag.Value)
  * @param transform - the name of a case-folding transform (CaseTransform.Value)
  */
sealed case class BagOfWordsConfig(accept: List[String], transform: String)
