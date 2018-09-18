package sat.dpll

import math._
import scala.io.Source
import scala.annotation._

object CNF {
  type Assgn = Map[Int, Boolean]

  case class Var(x: Int)

  case class Clause(xs: List[Int]) {
    def size = xs.size
    def contains(v: Int): Boolean = xs.contains(v)
    def remove(v: Int): Clause = Clause(xs.filter(_ != v))
    def containsAny(vs: List[Int]): Boolean = {
      for (v <- vs) { if (xs.contains(v)) return true }
      return false
    }
    def removeAllOccur(vs: List[Int]): Clause = Clause(xs.filter(!vs.contains(_)))

    def assign(v: Int, b: Boolean): Option[Clause] = {
      var new_xs = List[Int]()
      for (x <- xs) {
        if (abs(x) == abs(v)) { if ((x > 0) == b) return None }
        else { new_xs = x::new_xs }
      }
      Some(Clause(new_xs))
    }
  }

  case class Formula(cs: List[Clause]) {
    def isEmpty: Boolean = cs.isEmpty
    def containsMtClause: Boolean = cs.contains(Clause(List()))

    private def varsToAssignment(vars: Iterable[Int]) =
      vars.map((x:Int) => if (x>0) (x→true) else (-x→false)).toMap

    lazy private val unitVars = cs.filter(_.size == 1).map(_.xs.head)

    def containsUnit: Boolean = unitVars.size != 0

    def elimUnit(): (Formula, Assgn) = {
      val assign = varsToAssignment(unitVars)
      val result = for { c <- cs if !c.containsAny(unitVars) }
                   yield c.removeAllOccur(unitVars.map(-_))
      (Formula(result), assign)
    }

    lazy private val pureVars = cs.flatMap(_.xs).groupBy(abs).mapValues(_.size).filter({ case (k,v) => v == 1 }).keys.toList

    def containsPure: Boolean = pureVars.size != 0

    def elimPure(): (Formula, Assgn) = {
      val assign = varsToAssignment(pureVars)
      val result = for {c <- cs } yield c.removeAllOccur(pureVars)
      (Formula(result), assign)
    }

    def pick: Int = cs.head.xs.head
    def assign(v: Int, b: Boolean): Formula = {
      Formula((for (c <- cs) yield c.assign(v, b)).filter(_.nonEmpty).map(_.get))
    }
  }

  def parseLines(lines: Iterator[String]): Formula = {
    val cs = for (line <- lines if
                  !(line.startsWith("c") || line.startsWith("p") ||
                    line.startsWith("0") || line.startsWith("%") || line.isEmpty)) yield {
      val ns = line.split(" ").filter(_.nonEmpty).map(_.toInt).toList
      assert(ns.last == 0)
      Clause(ns.dropRight(1))
    }
    Formula(cs.toList)
  }

  def parse(input: String): Formula = parseLines(input.split("\\r?\\n").iterator)
}

object CNF_Examples {
  import CNF._
  /** (x1 ∨ x2 ∨ -x3 ∨ x6) ∧
    * (-x2 ∨ x4) ∧
    * (-x1 ∨ -x5) ∧
    * (-x1 ∨ x2 ∨ x3 ∨ -x4 ∨ x5) ∧
    * (-x3) ∧ (x2 ∨ x5) ∧
    * (-x5) ∧ (x1)
    */
  val example_f = Formula(List(Clause(List(1, 2, -3, 6)),
                               Clause(List(-2, 4)),
                               Clause(List(-1, -5)),
                               Clause(List(-1, 2, 3, -4, 5)),
                               Clause(List(-3)), Clause(List(2, 5)),
                               Clause(List(-5)), Clause(List(1))))
}

object DPLL {
  import CNF._

  def dpll(f: Formula, assgn: Map[Int, Boolean]): Option[Map[Int, Boolean]] = {
    if (f.containsMtClause) return None
    if (f.isEmpty) return Some(assgn)
    if (f.containsUnit) {
      val (new_f, new_assgn) = f.elimUnit
      return dpll(new_f, assgn ++ new_assgn)
    }
    if (f.containsPure) {
      val (new_f, new_assgn) = f.elimPure
      return dpll(new_f, assgn ++ new_assgn)
    }
    val v = f.pick
    val tryTrue = dpll(f.assign(v, true), assgn + (v → true))
    if (tryTrue.nonEmpty) tryTrue
    else dpll(f.assign(v, false), assgn + (v → false))
  }

  def solve(f: Formula): Option[Map[Int, Boolean]] = dpll(f, Map[Int, Boolean]()) match {
    case Some(m) => Some(m.map({ case (v, b) => if (v < 0) (-v, !b) else (v, b) }))
    case None => None
  }

}

object DPLLTest extends App {
  import CNF._
  import DPLL._
  import CNF_Examples._

  println("DPLL")
  println(example_f.elimUnit)
  println(example_f.elimPure)
  println(example_f.assign(6, true))
  println(example_f.assign(6, false))
  println(solve(example_f))

  /*
  val cnf_src1 = Source.fromResource("uuf200-860/uuf200-01.cnf").getLines //UNSAT
  val cnf1 = parseLines(cnf_src1)
  println(solve(cnf1))
   */

  val cnf_src1 = Source.fromResource("uf20-91/uf20-010.cnf").getLines //UNSAT
  val cnf1 = parseLines(cnf_src1)
  println(solve(cnf1))
}
