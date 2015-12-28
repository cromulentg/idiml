import com.ibm.icu.text.{BreakIterator}
import com.ibm.icu.util.ULocale
import java.lang.ThreadLocal
import java.lang.ref.SoftReference
import java.text.StringCharacterIterator

import scala.collection.mutable.{HashMap => MutableMap, Queue}
import scala.util.Try

import com.idibon.ml.cld2._

package com.idibon.ml.feature.tokenizer {

  /** Builder for ICUTokenizer instances, cache for BreakIterators
    *
    * Creating BreakIterator instances is expensive and slow (per ICU
    * documentation), so re-using iterators as broadly as possible is
    * highly desirable.
    */
  private[tokenizer] object ICUTokenizer {

    /** Tokenizes the text
      *
      * Auto-detects the locale for the provided text, then tokenizes
      * using any language-specific rules for that locale.
      *
      * Defaults to using the US if no locale is detected.
      */
    def tokenize(content: String): Seq[Token] = {
      // FIXME: detect HTML vs plain-text
      tokenize(content,
        identifyLocale(content, CLD2.DocumentMode.PlainText)
          .getOrElse(ULocale.US))
    }

    def tokenize(content: String, locale: ULocale): Seq[Token] = {
      breaking(locale, (breakIt: BreakIterator) => {
        breakIt.setText(content)

        // grab all boundaries in the document
        val boundaries = (breakIt.first ::
          Stream.continually(breakIt.next)
          .takeWhile(_ != BreakIterator.DONE).toList)

        /* iterate through neighboring pairs of segment boundaries to
         * generate the tokens */
        boundaries.zip(boundaries.tail).map({ case (first, last) => {
          val text = content.substring(first, last)
          new Token(text, Tag.of(text), first, last - first)
        }})
      })
    }

    /** Determines the primary locale for the document
      *
      * Uses language detection to assign a suitable locale for document
      * tokenization, returning None if no locale is found.
      *
      * @param content text to process
      * @param documentMode process the text in HTML or plain text mode
      * @return the primary locale for the document, or None if detection fails
      */
    private[tokenizer] def identifyLocale(content: String,
      documentMode: CLD2.DocumentMode): Option[ULocale] = {

      /* if CLD2 initialization fails, or detection throws an exception for
       * some reason, return None */
      Try { CLD2.detect(content, documentMode) match {
        case LangID.ENGLISH => Some(ULocale.US)
        case LangID.CHINESE => Some(ULocale.CHINA)
        case LangID.CHINESE_T => Some(ULocale.TAIWAN)
        case LangID.JAPANESE => Some(ULocale.JAPAN)
        case LangID.KOREAN => Some(ULocale.KOREA)
        case LangID.FRENCH => Some(ULocale.FRANCE)
        case LangID.GERMAN => Some(ULocale.GERMANY)
        case LangID.ITALIAN => Some(ULocale.ITALY)
        case _ => None
      } }.getOrElse(None)
    }

    /** Calls a user-provided function with a break iterator
      *
      * Allocates a word break iterator for use by the provided function
      * to segment text into words.
      *
      * @param locale locale of the text that will be analyzed.
      * @param fn callback function that accepts a break iterator as an
      *   argument
      * @return the return value from fn
      */
    private[tokenizer] def breaking[T](locale: ULocale, fn: (BreakIterator) => T): T = {
      /* grab the break iterator cache for this locale, allocate a new
       * cache if this is the first time that the locale is used */
      val localeCache = _breakIterators.synchronized {
        _breakIterators.get(locale).getOrElse({
          /* the first time we see a new locale, create a FIFO for caching
           * break iterators for that locale */
          val empty = Queue[SoftReference[BreakIterator]]()
          _breakIterators.put(locale, empty)
          empty
        })
      }

      /* grab a break iterator from the cache, if one exists, or
       * allocate a new iterator if not. */
      val iterator = localeCache.synchronized {
        try {
          /* soft references may be freed at any time, so loop over
           * cache entries until we find a live object */
          Stream.continually(localeCache.dequeue.get).find(_ != null)
        } catch {
          case _: NoSuchElementException => None
        }
      }.getOrElse({ BreakIterator.getWordInstance(locale) })

      try {
        fn(iterator)
      } finally {
        // return the iterator to the cache so it may be re-used
        localeCache.synchronized {
          localeCache.enqueue(new SoftReference(iterator))
        }
      }
    }

    private [this] val _breakIterators =
      MutableMap[ULocale, Queue[SoftReference[BreakIterator]]]()
  }
}
