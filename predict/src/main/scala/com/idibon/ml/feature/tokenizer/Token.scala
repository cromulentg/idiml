package com.idibon.ml.feature.tokenizer

import com.idibon.ml.alloy.Codec
import com.idibon.ml.feature._
import com.idibon.ml.feature.tokenizer.Unicode._

import com.ibm.icu.lang.UCharacter

/** Used to specify the types of tokens that may be generated
  *
  * Every token is assigned a type during tokenization (whitespace,
  * word, punctuation, etc.).
  *
  * NB: Tokens serialize tags as integer indices, so only add new values
  * to the end of the list!
  */
object Tag extends Enumeration {
  val Word,
    Punctuation,
    Whitespace,
    Markup,
    URI = Value

  /** Returns an ICU rule status value for the tag
    *
    * This value can be returned by a BreakIterator's getRuleStatus method,
    * to indicate the type of token that was segmented. Rule status values
    * are all negative, so as to not conflict with status tags defined
    * in the base ICU ruleset.
    *
    * {@link com.ibm.icu.text.BreakIterator#getRuleStatus}
    */
  def ruleStatus(v: Tag.Value) = 0 - (1 + v.id)

  /** Return the Tag (type) of the provided token content.
    *
    * @param content the content within the token
    * @param ruleStatus the rule status tag
    * @return the Tag of the token
    */
  def of(content: String, ruleStatus: Int) = {
    if (ruleStatus < 0) {
      Tag(-ruleStatus - 1)
    } else {
      var i = 0
      var whitespace = true
      var punctuation = true

      while (i < content.length && (whitespace || punctuation)) {
        val cp = content.codePointAt(i)
        i += (if (Character.isSupplementaryCodePoint(cp)) 2 else 1)

        whitespace &= UCharacter.isUWhiteSpace(cp)
        punctuation &= (isPunctuation(cp) || isGraphemeExtender(cp))
      }

      if (whitespace)
        Whitespace
      else if (punctuation)
        Punctuation
      else
        Word
    }
  }
}

case class Token(content: String, tag: Tag.Value, offset: Int, length: Int)
    extends Feature[Token] with Buildable[Token, TokenBuilder] {

  final val end = offset + length

  def get = this

  def save(output: FeatureOutputStream) {
    Codec.String.write(output, content)
    Codec.VLuint.write(output, tag.id)
    Codec.VLuint.write(output, offset)
    Codec.VLuint.write(output, length)
  }

  def getHumanReadableString: Option[String] = {
    Some(s"${this.content} (${this.offset}, ${this.length})")
  }
}

class TokenBuilder extends Builder[Token] {
  def build(input: FeatureInputStream) = {
    new Token(Codec.String.read(input),
      Tag.apply(Codec.VLuint.read(input)),
      Codec.VLuint.read(input),
      Codec.VLuint.read(input))
  }
}

