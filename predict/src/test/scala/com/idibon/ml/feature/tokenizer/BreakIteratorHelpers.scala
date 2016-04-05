package com.idibon.ml.feature.tokenizer

import java.lang.ThreadLocal

import com.ibm.icu

/** Helper methods for break iterator specs */
trait BreakIteratorHelpers[T <: icu.text.BreakIterator] {

  def newIterator(delegate: icu.text.BreakIterator): T

  val breakIterator = new ThreadLocal[T]() {
    override def initialValue() = newIterator(
      icu.text.BreakIterator.getWordInstance(icu.util.ULocale.US))
  }

  def !!(x: String): Seq[String] = {
    val i = breakIterator.get
    i.setText(x)
    val boundaries = (i.first ::
      Stream.continually(i.next)
      .takeWhile(_ != icu.text.BreakIterator.DONE).toList)
    boundaries.sliding(2).map(tok => x.substring(tok.head, tok.last)).toSeq
  }

  def tag(x: String): Seq[(String, Int)] = {
    val i = breakIterator.get
    i.setText(x)
    val boundaries = ((i.first, 0) ::
      Stream.continually((i.next, i.getRuleStatus))
      .takeWhile(_._1 != icu.text.BreakIterator.DONE).toList)
    boundaries.sliding(2).map(tok => {
      (x.substring(tok.head._1, tok.last._1), tok.last._2)
    }).toSeq
  }
}
