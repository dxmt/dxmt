use crate::dxql::compiler::{FilterExpr, QueryPlan, SelectPlan};
use crate::dxql::DxqlError;
use std::collections::HashSet;

#[derive(Debug, Clone)]
pub struct Optimizer {
    _private: (),
}

impl Default for Optimizer {
    fn default() -> Self {
        Self::new()
    }
}

impl Optimizer {
    pub fn new() -> Self {
        Self { _private: () }
    }

    pub fn optimize(&self, plan: QueryPlan) -> Result<QueryPlan, DxqlError> {
        match plan {
            QueryPlan::Select(select) => Ok(QueryPlan::Select(optimize_select(select))),
            _ => Ok(plan),
        }
    }
}

fn optimize_select(mut select: SelectPlan) -> SelectPlan {
    if let Some(filter) = select.filter.as_mut() {
        order_conjuncts(filter);
    }
    dedup_projections(&mut select);
    select
}

/// Within an AND, evaluate cheaper predicates first; the executor short-circuits, so a failing
/// equality spares the LIKE after it. Reordering a conjunction cannot change which rows match,
/// and OR branches keep their order.
fn order_conjuncts(expr: &mut FilterExpr) {
    match expr {
        FilterExpr::And(terms) => {
            terms.iter_mut().for_each(order_conjuncts);
            terms.sort_by_key(cost);
        }
        FilterExpr::Or(terms) => terms.iter_mut().for_each(order_conjuncts),
        FilterExpr::Compare(_) | FilterExpr::In { .. } | FilterExpr::Between { .. } => {}
    }
}

fn cost(expr: &FilterExpr) -> u8 {
    match expr {
        FilterExpr::Compare(filter) => match filter.operation.as_str() {
            "=" => 0,
            "!=" => 1,
            ">" | ">=" | "<" | "<=" => 2,
            "LIKE" => 3,
            _ => 4,
        },
        FilterExpr::In { .. } => 1,
        FilterExpr::Between { .. } => 2,
        FilterExpr::And(_) | FilterExpr::Or(_) => 5,
    }
}

/// Drop a projection that repeats an earlier output name. `rank AS a, rank AS b` names two
/// outputs and keeps both.
fn dedup_projections(select: &mut SelectPlan) {
    let mut seen = HashSet::new();
    select
        .projections
        .retain(|p| seen.insert(p.alias.clone().unwrap_or_else(|| p.field.clone())));
}
