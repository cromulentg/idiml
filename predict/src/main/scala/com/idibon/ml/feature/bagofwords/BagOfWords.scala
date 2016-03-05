package com.idibon.ml.feature.bagofwords

import com.idibon.ml.feature.{Feature, FeatureTransformer}
import com.idibon.ml.feature.tokenizer.{Token, Tag}
import com.idibon.ml.feature.language.LanguageCode
import com.idibon.ml.common.{Archivable, ArchiveLoader, Engine}
import com.idibon.ml.alloy.Alloy
import org.json4s._
import org.json4s.JsonDSL._

/** BagOfWords FeatureTransformer
  *
  * Produces a bag of words. i.e. a Word Feature for each token, optionally
  * applying language-specific case folding and filtering by the token type
  *
  * @param accept - a list of accepted (i.e., converted) token types
  * @param transform - a case folding operation to apply
  */
class BagOfWordsTransformer(accept: Seq[Tag.Value],
  val transform: CaseTransform.Value)
    extends FeatureTransformer with CaseTransform
    with Archivable[BagOfWordsTransformer, BagOfWordsTransformerLoader] {

  // for performance, convert the list of accepted token tags into a bitmask
  private[this] val _tokenMask = accept.map(_.id)
    .foldLeft(0)((accum, id) => accum | (1 << id))

  /** Produces a bag of words from a sequence of Tokens
    *
    * @param toks - the generated tokens
    * @param lc - the primary language identified for the token stream
    * @return bag of words represented as a sequence of Word features.
    */
  def apply(toks: Seq[Feature[Token]], lc: Feature[LanguageCode]): Seq[Word] = {
    val accepted = toks.filter(t => (_tokenMask & (1 << t.get.tag.id)) != 0)
    val f = transformation(lc)
    accepted.map(t => f(t.get))
  }

  /** Saves the configuration data for this transform to an Alloy */
  def save(writer: Alloy.Writer): Option[JObject] = {
    Some(("accept" -> this.accept.map(_.toString)) ~
      ("transform" -> this.transform.toString))
  }
}

/** Paired loader for BagOfWordsTransformer */
class BagOfWordsTransformerLoader extends ArchiveLoader[BagOfWordsTransformer] {
  def load(engine: Engine, reader: Option[Alloy.Reader], config: Option[JObject]) = {
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
