package com.idibon.ml.feature.tokenizer

import com.ibm.icu.util.CharsTrie
import com.ibm.icu.text.UCharacterIterator

/** Matches a CharacterIterator against a Trie of known values
  *
  * If a match is detected at the CharacterIterator's current position,
  * returns an enumerated value corresponding to the type of match detected
  *
  * Trie instances may only be used by a single thread.
  */
private[tokenizer] class Trie[T <: Enumeration](values: T, trie: CharsTrie) {

  /** Shallow-copies the current trie to allow for parallel evaluation */
  override def clone: Trie[T] =
    new Trie[T](values, trie.clone.asInstanceOf[CharsTrie])

  /** Compares the current text position against values in the trie
    *
    * Returns the type and length of the matched text if a match is detected.
    *
    * @param text iterator to analyze the current text boundary
    * @return match result, if a match is detected
    */
  def matches(text: UCharacterIterator): Option[(T#Value, Int)] = {
    trie.reset()
    val x: Option[(T#Value, Int)] = None
    Stream.continually(text.nextCodePoint)
      .takeWhile(u => trie.nextForCodePoint(u).hasNext())
      .foldLeft(x)({ case (best, codePoint) => trie.current.hasValue() match {
        case true => Some(values(trie.getValue()), text.getIndex())
        case _ => best
      }})
  }
}
