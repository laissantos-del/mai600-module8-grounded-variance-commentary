# Synthetic corpus

Every document here is invented. No real company's figures, policies, or internal memos appear anywhere in this project. The documents are written to mirror the artefacts a mid-size manufacturer actually produces at close, so that retrieval faces the same difficulties it would face in production: several versions of the same policy, memos that were published after the close they would explain, and near-duplicate memos about the wrong entity.

The retrieval `category` is what splits the corpus into the two axes.

| category | axis | what it is |
|---|---|---|
| `rules` | rules axis | policy, materiality thresholds, ownership |
| `causes` | causes axis | driver memos that explain one variance |
| `precedent` | excluded | prior-period commentary packs |
| `distractor` | causes axis | plausible memos about the wrong scope or period |

`corpus_documents.json` is the machine-readable source of truth. These Markdown files are generated from it so a finance reviewer can read them directly.

## Index

| doc | title | category | published | file |
|---|---|---|---|---|
| D01 | Variance Commentary Policy | `rules` | 2022-12-15 | [D01_variance_commentary_policy.md](D01_variance_commentary_policy.md) |
| D03 | KPI and Metric Definitions Dictionary | `rules` | 2023-01-05 | [D03_kpi_and_metric_definitions_dictionary.md](D03_kpi_and_metric_definitions_dictionary.md) |
| D04 | Cost Centre Owner Register | `rules` | 2023-01-03 | [D04_cost_centre_owner_register.md](D04_cost_centre_owner_register.md) |
| D11 | Procurement Memo — Steel Index Movement | `causes` | 2023-06-12 | [D11_procurement_memo_steel_index_movement.md](D11_procurement_memo_steel_index_movement.md) |
| D12 | Sales Operations Memo — Tier-1 OEM Contract Non-Renewal | `causes` | 2023-11-08 | [D12_sales_operations_memo_tier_1_oem_contract_non_renewal.md](D12_sales_operations_memo_tier_1_oem_contract_non_renewal.md) |
| D13 | Pricing Action Memo (v1.0, PROPOSED) | `causes` | 2024-01-10 | [D13_pricing_action_memo_v1_0_proposed.md](D13_pricing_action_memo_v1_0_proposed.md) |
| D14 | Pricing Action Memo (v2.0, APPROVED) | `causes` | 2024-01-25 | [D14_pricing_action_memo_v2_0_approved.md](D14_pricing_action_memo_v2_0_approved.md) |
| D15 | R&D Headcount Plan FY2024 | `causes` | 2024-05-06 | [D15_r_d_headcount_plan_fy2024.md](D15_r_d_headcount_plan_fy2024.md) |
| D16 | Treasury Memo — EUR Exposure Update | `causes` | 2024-12-09 | [D16_treasury_memo_eur_exposure_update.md](D16_treasury_memo_eur_exposure_update.md) |
| D17 | Treasury Memo — Revolver Draw and Leverage Grid | `causes` | 2025-03-11 | [D17_treasury_memo_revolver_draw_and_leverage_grid.md](D17_treasury_memo_revolver_draw_and_leverage_grid.md) |
| D18 | Aftermarket Launch Brief | `causes` | 2025-06-18 | [D18_aftermarket_launch_brief.md](D18_aftermarket_launch_brief.md) |
| D19 | Operations Memo — Inventory Write-Down | `causes` | 2025-10-09 | [D19_operations_memo_inventory_write_down.md](D19_operations_memo_inventory_write_down.md) |
| D20 | Monthly Management Commentary Pack — June 2023 | `precedent` | 2023-07-05 | [D20_monthly_management_commentary_pack_june_2023.md](D20_monthly_management_commentary_pack_june_2023.md) |
| D32 | Procurement Memo — Resin Cost Movement | `distractor` | 2023-06-14 | [D32_procurement_memo_resin_cost_movement.md](D32_procurement_memo_resin_cost_movement.md) |
| D34 | Sales Headcount Plan FY2024 | `distractor` | 2024-05-05 | [D34_sales_headcount_plan_fy2024.md](D34_sales_headcount_plan_fy2024.md) |
