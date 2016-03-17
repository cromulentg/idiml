package com.idibon.ml.feature.tokenizer

import com.ibm.icu.lang.UCharacter
import com.ibm.icu.lang.UCharacterEnums.ECharacterCategory._
import com.ibm.icu.lang.UProperty._

/** Helper methods for processing Unicode code points */
package object Unicode {

  /** Returns true if the code point is a punctuation mark.
    *
    * The list of unicode character categories that are mapped onto the
    * "punctuation" test is taken from
    *  http://icu-project.org/apiref/icu4j/com/ibm/icu/lang/UCharacter.html
    *
    * @param ch Unicode codepoint
    * @return true for punctuation marks, else false
    */
  def isPunctuation(ch: Int): Boolean = {
    UCharacter.getType(ch) match {
      case DASH_PUNCTUATION
         | START_PUNCTUATION
         | END_PUNCTUATION
         | CONNECTOR_PUNCTUATION
         | OTHER_PUNCTUATION
         | INITIAL_PUNCTUATION
         | FINAL_PUNCTUATION => true
      case _ => false
    }
  }

  /** Returns true if the code point is a grapheme extender
    *
    * Grapheme extenders include combining diacritical marks, variation
    * selectors, and other codepoints that should be included in the same
    * grapheme as the previous base codepoint.
    *
    * @param ch Unicode codepoint
    * @return true if the codepoint is a grapheme extender, else false
    */
  def isGraphemeExtender(ch: Int): Boolean =
    UCharacter.hasBinaryProperty(ch, GRAPHEME_EXTEND)

  /** Returns true if the codepoint is a hexadecimal digit
    *
    * @param ch Unicode codepoint
    * @return true if the codepoint is ASCII and a hex digit, else false
    */
  def isAsciiHexDigit(ch: Int): Boolean = {
    ((ch >= '0' && ch <= '9') ||
      (ch >= 'a' && ch <= 'f') ||
      (ch >= 'A' && ch <= 'F'))
  }

  /** Returns true if the codepoint is an ASCII digit
    *
    * @param ch Unicode codepoint
    * @return true if the codepoint is an ASCII digit, else false
    */
  def isAsciiDigit(ch: Int): Boolean = (ch >= '0' && ch <= '9')
}
