use crate::dxql::{DxqlError, compiler::QueryPlan};

#[derive(Debug, Clone)]
pub struct Optimizer {
    _private: (),
}

impl Optimizer {
    pub fn new() -> Self {
        Self { _private: () }
    }

    pub fn optimize(&self, plan: QueryPlan) -> Result<QueryPlan, DxqlError> {
        match plan {
            QueryPlan::Select(select) => self.optimize_select(select),
            _ => Ok(plan),
        }
    }

    fn optimize_select(&self, mut select: crate::dxql::compiler::SelectPlan) -> Result<QueryPlan, DxqlError> {
        self.push_down_filters(&mut select);

        self.optimize_projections(&mut select);

        Ok(QueryPlan::Select(select))
    }

    /// Reorder filters to evaluate cheaper predicates first.
    /// Priority: equality (=) > inequality (!=) > range (>, >=, <, <=) > LIKE
    fn push_down_filters(&self, select: &mut crate::dxql::compiler::SelectPlan) {
        fn filter_cost(op: &str) -> u8 {
            match op {
                "=" | "==" | "eq" => 0,
                "!=" | "ne" => 1,
                ">" | ">=" | "<" | "<=" => 2,
                "LIKE" | "like" => 3,
                _ => 4,
            }
        }
        select.filters.sort_by_key(|f| filter_cost(&f.operation));
    }

    /// Remove duplicate projections and normalize wildcard.
    fn optimize_projections(&self, select: &mut crate::dxql::compiler::SelectPlan) {
        // If any projection is "*", keep only "*" (fetch all fields)
        if select.projections.iter().any(|p| p.field == "*") {
            select.projections.retain(|p| p.field == "*");
            select.projections.dedup_by(|a, b| a.field == b.field);
            return;
        }
        // Deduplicate by field name, keeping first occurrence
        let mut seen = std::collections::HashSet::new();
        select.projections.retain(|p| seen.insert(p.field.clone()));
    }
}