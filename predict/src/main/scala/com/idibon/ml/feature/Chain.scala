package com.idibon.ml.feature

import scala.collection.{GenTraversable, AbstractTraversable}

/** Immutable, ordered collection of links of type T, traversable as a
  * doubly-linked list of ChainLinks
  */
class Chain[+T](private[feature] val nodes: IndexedSeq[T])
    extends AbstractTraversable[ChainLink[T]] {

  override def foreach[U](f: (ChainLink[T]) => U) {
    // lazily instantiate Links, to minimize the number of reachable objects
    (0 until nodes.size).foreach(i => f(new ChainLink(this, i)))
  }
}

/** Represents a single link in a Chain
  *
  * Provides an indirect accessor to the original item, and methods to
  * scan to adjacent items.
  */
sealed class ChainLink[+T](chain: Chain[T], index: Int) {

  /** Returns the item at the current link */
  def value = chain.nodes(index)

  /** Returns the next link in the chain, if one exists */
  def next: Option[ChainLink[T]] = {
    if (chain.nodes.isDefinedAt(index + 1))
      Some(new ChainLink(chain, index + 1))
    else
      None
  }

  /** Returns the previous link in the chain, if one exists */
  def previous: Option[ChainLink[T]] = {
    if (chain.nodes.isDefinedAt(index - 1))
      Some(new ChainLink(chain, index - 1))
    else
      None
  }
}

/** Chain companion object */
object Chain {
  /** Chain builder for any collection which has a definite size. */
  def apply[T](nodes: GenTraversable[T]) = nodes match {
    case x: IndexedSeq[T] => new Chain(x)
    case x if x.hasDefiniteSize  => new Chain(x.toVector)
    case _ => throw new UnsupportedOperationException("Infinite collection")
  }
}
