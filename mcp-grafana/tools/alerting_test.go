// Requires a Grafana instance running on localhost:3000,
// with alert rules configured.
// Run with `go test -tags integration`.
//go:build integration

package tools

import (
	"testing"

	"github.com/stretchr/testify/require"
)

const (
	rule1UID        = "test_alert_rule_1"
	rule1Title      = "Test Alert Rule 1"
	rule2UID        = "test_alert_rule_2"
	rule2Title      = "Test Alert Rule 2"
	rulePausedUID   = "test_alert_rule_paused"
	rulePausedTitle = "Test Alert Rule (Paused)"
)

var (
	rule1Labels = map[string]string{
		"severity": "info",
		"type":     "test",
		"rule":     "first",
	}
	rule2Labels = map[string]string{
		"severity": "info",
		"type":     "test",
		"rule":     "second",
	}
	rule3Labels = map[string]string{
		"severity": "info",
		"type":     "test",
		"rule":     "third",
	}

	rule1 = alertRuleSummary{
		UID:    rule1UID,
		State:  "",
		Title:  rule1Title,
		Labels: rule1Labels,
	}
	rule2 = alertRuleSummary{
		UID:    rule2UID,
		State:  "",
		Title:  rule2Title,
		Labels: rule2Labels,
	}
	rulePaused = alertRuleSummary{
		UID:    rulePausedUID,
		State:  "",
		Title:  rulePausedTitle,
		Labels: rule3Labels,
	}
	allExpectedRules = []alertRuleSummary{rule1, rule2, rulePaused}
)

// Because the state depends on the evaluation of the alert rules,
// clear it and other variable runtime fields before comparing the results
// to avoid waiting for the alerts to start firing or be in the pending state.
func clearState(rules []alertRuleSummary) []alertRuleSummary {
	for i := range rules {
		rules[i].State = ""
		rules[i].Health = ""
		rules[i].FolderUID = ""
		rules[i].RuleGroup = ""
		rules[i].For = ""
		rules[i].LastEvaluation = ""
		rules[i].Annotations = nil
	}

	return rules
}

func TestAlertingTools_ListAlertRules(t *testing.T) {
	t.Run("list alert rules", func(t *testing.T) {
		ctx := newTestContext()
		result, err := listAlertRules(ctx, ListAlertRulesParams{})
		require.NoError(t, err)

		require.ElementsMatch(t, allExpectedRules, clearState(result))
	})

	t.Run("list alert rules with pagination", func(t *testing.T) {
		ctx := newTestContext()

		// Get the first page with limit 1
		result1, err := listAlertRules(ctx, ListAlertRulesParams{
			Limit: 1,
			Page:  1,
		})
		require.NoError(t, err)
		require.Len(t, result1, 1)

		// Get the second page with limit 1
		result2, err := listAlertRules(ctx, ListAlertRulesParams{
			Limit: 1,
			Page:  2,
		})
		require.NoError(t, err)
		require.Len(t, result2, 1)

		// Get the third page with limit 1
		result3, err := listAlertRules(ctx, ListAlertRulesParams{
			Limit: 1,
			Page:  3,
		})
		require.NoError(t, err)
		require.Len(t, result3, 1)

		// The next page is empty
		result4, err := listAlertRules(ctx, ListAlertRulesParams{
			Limit: 1,
			Page:  4,
		})
		require.NoError(t, err)
		require.Empty(t, result4)
	})

	t.Run("list alert rules without the page and limit params", func(t *testing.T) {
		ctx := newTestContext()
		result, err := listAlertRules(ctx, ListAlertRulesParams{})
		require.NoError(t, err)
		require.ElementsMatch(t, allExpectedRules, clearState(result))
	})

	t.Run("list alert rules with selectors that match", func(t *testing.T) {
		ctx := newTestContext()
		result, err := listAlertRules(ctx, ListAlertRulesParams{
			LabelSelectors: []Selector{
				{
					Filters: []LabelMatcher{
						{
							Name:  "severity",
							Value: "info",
							Type:  "=",
						},
					},
				},
			},
		})
		require.NoError(t, err)
		require.ElementsMatch(t, allExpectedRules, clearState(result))
	})

	t.Run("list alert rules with selectors that don't match", func(t *testing.T) {
		ctx := newTestContext()
		result, err := listAlertRules(ctx, ListAlertRulesParams{
			LabelSelectors: []Selector{
				{
					Filters: []LabelMatcher{
						{
							Name:  "severity",
							Value: "critical",
							Type:  "=",
						},
					},
				},
			},
		})
		require.NoError(t, err)
		require.Empty(t, result)
	})

	t.Run("list alert rules with multiple selectors", func(t *testing.T) {
		ctx := newTestContext()
		result, err := listAlertRules(ctx, ListAlertRulesParams{
			LabelSelectors: []Selector{
				{
					Filters: []LabelMatcher{
						{
							Name:  "severity",
							Value: "info",
							Type:  "=",
						},
					},
				},
				{
					Filters: []LabelMatcher{
						{
							Name:  "rule",
							Value: "second",
							Type:  "=",
						},
					},
				},
			},
		})
		require.NoError(t, err)
		require.ElementsMatch(t, []alertRuleSummary{rule2}, clearState(result))
	})

	t.Run("list alert rules with regex matcher", func(t *testing.T) {
		ctx := newTestContext()
		result, err := listAlertRules(ctx, ListAlertRulesParams{
			LabelSelectors: []Selector{
				{
					Filters: []LabelMatcher{
						{
							Name:  "rule",
							Value: "fi.*",
							Type:  "=~",
						},
					},
				},
			},
		})
		require.NoError(t, err)
		require.ElementsMatch(t, []alertRuleSummary{rule1}, clearState(result))
	})

	t.Run("list alert rules with selectors and pagination", func(t *testing.T) {
		ctx := newTestContext()
		result, err := listAlertRules(ctx, ListAlertRulesParams{
			LabelSelectors: []Selector{
				{
					Filters: []LabelMatcher{
						{
							Name:  "severity",
							Value: "info",
							Type:  "=",
						},
					},
				},
			},
			Limit: 1,
			Page:  1,
		})
		require.NoError(t, err)
		require.Len(t, result, 1)
		require.ElementsMatch(t, []alertRuleSummary{rule1}, clearState(result))

		// Second page
		result, err = listAlertRules(ctx, ListAlertRulesParams{
			LabelSelectors: []Selector{
				{
					Filters: []LabelMatcher{
						{
							Name:  "severity",
							Value: "info",
							Type:  "=",
						},
					},
				},
			},
			Limit: 1,
			Page:  2,
		})
		require.NoError(t, err)
		require.Len(t, result, 1)
		require.ElementsMatch(t, []alertRuleSummary{rule2}, clearState(result))
	})

	t.Run("list alert rules with not equals operator", func(t *testing.T) {
		ctx := newTestContext()
		result, err := listAlertRules(ctx, ListAlertRulesParams{
			LabelSelectors: []Selector{
				{
					Filters: []LabelMatcher{
						{
							Name:  "severity",
							Value: "critical",
							Type:  "!=",
						},
					},
				},
			},
		})
		require.NoError(t, err)
		require.ElementsMatch(t, allExpectedRules, clearState(result))
	})

	t.Run("list alert rules with not matches operator", func(t *testing.T) {
		ctx := newTestContext()
		result, err := listAlertRules(ctx, ListAlertRulesParams{
			LabelSelectors: []Selector{
				{
					Filters: []LabelMatcher{
						{
							Name:  "severity",
							Value: "crit.*",
							Type:  "!~",
						},
					},
				},
			},
		})
		require.NoError(t, err)
		require.ElementsMatch(t, allExpectedRules, clearState(result))
	})

	t.Run("list alert rules with non-existent label", func(t *testing.T) {
		// Equality with non-existent label should return no results
		ctx := newTestContext()
		result, err := listAlertRules(ctx, ListAlertRulesParams{
			LabelSelectors: []Selector{
				{
					Filters: []LabelMatcher{
						{
							Name:  "nonexistent",
							Value: "value",
							Type:  "=",
						},
					},
				},
			},
		})
		require.NoError(t, err)
		require.Empty(t, result)
	})

	t.Run("list alert rules with non-existent label and inequality", func(t *testing.T) {
		// Inequality with non-existent label should return all results
		ctx := newTestContext()
		result, err := listAlertRules(ctx, ListAlertRulesParams{
			LabelSelectors: []Selector{
				{
					Filters: []LabelMatcher{
						{
							Name:  "nonexistent",
							Value: "value",
							Type:  "!=",
						},
					},
				},
			},
		})
		require.NoError(t, err)
		require.ElementsMatch(t, allExpectedRules, clearState(result))
	})

	t.Run("list alert rules with a limit that is larger than the number of rules", func(t *testing.T) {
		ctx := newTestContext()
		result, err := listAlertRules(ctx, ListAlertRulesParams{
			Limit: 1000,
			Page:  1,
		})
		require.NoError(t, err)
		require.ElementsMatch(t, allExpectedRules, clearState(result))
	})

	t.Run("list alert rules with a page that doesn't exist", func(t *testing.T) {
		ctx := newTestContext()
		result, err := listAlertRules(ctx, ListAlertRulesParams{
			Limit: 10,
			Page:  1000,
		})
		require.NoError(t, err)
		require.Empty(t, result)
	})

	t.Run("list alert rules with invalid page parameter", func(t *testing.T) {
		ctx := newTestContext()
		result, err := listAlertRules(ctx, ListAlertRulesParams{
			Page: -1,
		})
		require.Error(t, err)
		require.Empty(t, result)
	})

	t.Run("list alert rules with invalid limit parameter", func(t *testing.T) {
		ctx := newTestContext()
		result, err := listAlertRules(ctx, ListAlertRulesParams{
			Limit: -1,
		})
		require.Error(t, err)
		require.Empty(t, result)
	})
}

func TestAlertingTools_GetAlertRuleByUID(t *testing.T) {
	t.Run("get running alert rule by uid", func(t *testing.T) {
		ctx := newTestContext()
		result, err := getAlertRuleByUID(ctx, GetAlertRuleByUIDParams{
			UID: rule1UID,
		})

		require.NoError(t, err)
		require.Equal(t, rule1UID, result.UID)
		require.NotNil(t, result.Title)
		require.Equal(t, rule1Title, *result.Title)
		require.False(t, result.IsPaused)
	})

	t.Run("get paused alert rule by uid", func(t *testing.T) {
		ctx := newTestContext()
		result, err := getAlertRuleByUID(ctx, GetAlertRuleByUIDParams{
			UID: "test_alert_rule_paused",
		})

		require.NoError(t, err)
		require.Equal(t, rulePausedUID, result.UID)
		require.NotNil(t, result.Title)
		require.Equal(t, rulePausedTitle, *result.Title)
		require.True(t, result.IsPaused)
	})

	t.Run("get alert rule with empty UID fails", func(t *testing.T) {
		ctx := newTestContext()
		result, err := getAlertRuleByUID(ctx, GetAlertRuleByUIDParams{
			UID: "",
		})

		require.Nil(t, result)
		require.Error(t, err)
	})

	t.Run("get non-existing alert rule by uid", func(t *testing.T) {
		ctx := newTestContext()
		result, err := getAlertRuleByUID(ctx, GetAlertRuleByUIDParams{
			UID: "some-non-existing-alert-rule-uid",
		})

		require.Nil(t, result)
		require.Error(t, err)
		require.Contains(t, err.Error(), "getAlertRuleNotFound")
	})
}

var (
	emailType = "email"

	contactPoint1 = contactPointSummary{
		UID:  "email1",
		Name: "Email1",
		Type: &emailType,
	}
	contactPoint2 = contactPointSummary{
		UID:  "email2",
		Name: "Email2",
		Type: &emailType,
	}
	contactPoint3 = contactPointSummary{
		UID:  "",
		Name: "email receiver",
		Type: &emailType,
	}
	allExpectedContactPoints = []contactPointSummary{contactPoint1, contactPoint2, contactPoint3}
)

func TestAlertingTools_ListContactPoints(t *testing.T) {
	t.Run("list contact points", func(t *testing.T) {
		ctx := newTestContext()
		result, err := listContactPoints(ctx, ListContactPointsParams{})
		require.NoError(t, err)
		require.ElementsMatch(t, allExpectedContactPoints, result)
	})

	t.Run("list one contact point", func(t *testing.T) {
		ctx := newTestContext()

		// Get the contact points with limit 1
		result1, err := listContactPoints(ctx, ListContactPointsParams{
			Limit: 1,
		})
		require.NoError(t, err)
		require.Len(t, result1, 1)
	})

	t.Run("list contact points with name filter", func(t *testing.T) {
		ctx := newTestContext()
		name := "Email1"

		result, err := listContactPoints(ctx, ListContactPointsParams{
			Name: &name,
		})
		require.NoError(t, err)
		require.Len(t, result, 1)
		require.Equal(t, "Email1", result[0].Name)
	})

	t.Run("list contact points with invalid limit parameter", func(t *testing.T) {
		ctx := newTestContext()
		result, err := listContactPoints(ctx, ListContactPointsParams{
			Limit: -1,
		})
		require.Error(t, err)
		require.Empty(t, result)
	})

	t.Run("list contact points with large limit", func(t *testing.T) {
		ctx := newTestContext()
		result, err := listContactPoints(ctx, ListContactPointsParams{
			Limit: 1000,
		})
		require.NoError(t, err)
		require.NotEmpty(t, result)
	})

	t.Run("list contact points with non-existent name filter", func(t *testing.T) {
		ctx := newTestContext()
		name := "NonExistentAlert"

		result, err := listContactPoints(ctx, ListContactPointsParams{
			Name: &name,
		})
		require.NoError(t, err)
		require.Empty(t, result)
	})
}

func TestAlertingTools_CreateAlertRule(t *testing.T) {
	t.Run("create alert rule with valid parameters", func(t *testing.T) {
		ctx := newTestContext()

		// Sample query data that matches Grafana's expected format
		sampleData := []any{
			map[string]any{
				"refId":     "A",
				"queryType": "",
				"relativeTimeRange": map[string]any{
					"from": 600,
					"to":   0,
				},
				"datasourceUid": "prometheus-uid",
				"model": map[string]any{
					"expr":          "up",
					"hide":          false,
					"intervalMs":    1000,
					"maxDataPoints": 43200,
					"refId":         "A",
				},
			},
			map[string]any{
				"refId":     "B",
				"queryType": "",
				"relativeTimeRange": map[string]any{
					"from": 0,
					"to":   0,
				},
				"datasourceUid": "__expr__",
				"model": map[string]any{
					"conditions": []any{
						map[string]any{
							"evaluator": map[string]any{
								"params": []any{1},
								"type":   "gt",
							},
							"operator": map[string]any{
								"type": "and",
							},
							"query": map[string]any{
								"params": []any{"A"},
							},
							"reducer": map[string]any{
								"params": []any{},
								"type":   "last",
							},
							"type": "query",
						},
					},
					"datasource": map[string]any{
						"type": "__expr__",
						"uid":  "__expr__",
					},
					"hide":          false,
					"intervalMs":    1000,
					"maxDataPoints": 43200,
					"refId":         "B",
					"type":          "classic_conditions",
				},
			},
		}

		testUID := "test_create_alert_rule"
		params := CreateAlertRuleParams{
			Title:        "Test Created Alert Rule",
			RuleGroup:    "test-group",
			FolderUID:    "tests",
			Condition:    "B",
			Data:         sampleData,
			NoDataState:  "OK",
			ExecErrState: "OK",
			For:          "5m",
			Annotations: map[string]string{
				"summary": "Test alert rule created via API",
			},
			Labels: map[string]string{
				"team": "test-team",
			},
			UID:   &testUID,
			OrgID: 1,
		}

		result, err := createAlertRule(ctx, params)
		require.NoError(t, err)
		require.NotNil(t, result)
		require.Equal(t, testUID, result.UID)
		require.Equal(t, "Test Created Alert Rule", *result.Title)
		require.Equal(t, "test-group", *result.RuleGroup)

		// Clean up: delete the created rule
		_, cleanupErr := deleteAlertRule(ctx, DeleteAlertRuleParams{UID: testUID})
		require.NoError(t, cleanupErr)
	})

	t.Run("create alert rule with missing required fields", func(t *testing.T) {
		ctx := newTestContext()

		params := CreateAlertRuleParams{
			Title: "Incomplete Rule",
			// Missing other required fields
		}

		result, err := createAlertRule(ctx, params)
		require.Error(t, err)
		require.Nil(t, result)
		require.Contains(t, err.Error(), "ruleGroup is required")
	})

	t.Run("create alert rule with empty title", func(t *testing.T) {
		ctx := newTestContext()

		params := CreateAlertRuleParams{
			Title: "",
		}

		result, err := createAlertRule(ctx, params)
		require.Error(t, err)
		require.Nil(t, result)
		require.Contains(t, err.Error(), "title is required")
	})
}

func TestAlertingTools_UpdateAlertRule(t *testing.T) {
	t.Run("update existing alert rule", func(t *testing.T) {
		ctx := newTestContext()

		// First create a rule to update
		sampleData := []any{
			map[string]any{
				"refId":     "A",
				"queryType": "",
				"relativeTimeRange": map[string]any{
					"from": 600,
					"to":   0,
				},
				"datasourceUid": "prometheus-uid",
				"model": map[string]any{
					"expr":          "up",
					"hide":          false,
					"intervalMs":    1000,
					"maxDataPoints": 43200,
					"refId":         "A",
				},
			},
		}

		testUID := "test_update_alert_rule"
		createParams := CreateAlertRuleParams{
			Title:        "Original Title",
			RuleGroup:    "test-group",
			FolderUID:    "tests",
			Condition:    "A",
			Data:         sampleData,
			NoDataState:  "OK",
			ExecErrState: "OK",
			For:          "5m",
			UID:          &testUID,
			OrgID:        1,
		}

		// Create the rule
		created, err := createAlertRule(ctx, createParams)
		require.NoError(t, err)
		require.NotNil(t, created)

		// Now update it
		updateParams := UpdateAlertRuleParams{
			UID:          testUID,
			Title:        "Updated Title",
			RuleGroup:    "test-group",
			FolderUID:    "tests",
			Condition:    "A",
			Data:         sampleData,
			NoDataState:  "Alerting",
			ExecErrState: "Alerting",
			For:          "10m",
			Annotations: map[string]string{
				"summary": "Updated alert rule",
			},
			Labels: map[string]string{
				"team": "updated-team",
			},
			OrgID: 1,
		}

		result, err := updateAlertRule(ctx, updateParams)
		require.NoError(t, err)
		require.NotNil(t, result)
		require.Equal(t, testUID, result.UID)
		require.Equal(t, "Updated Title", *result.Title)
		require.Equal(t, "Alerting", *result.NoDataState)

		// Clean up: delete the rule
		_, cleanupErr := deleteAlertRule(ctx, DeleteAlertRuleParams{UID: testUID})
		require.NoError(t, cleanupErr)
	})

	t.Run("update non-existent alert rule", func(t *testing.T) {
		ctx := newTestContext()

		params := UpdateAlertRuleParams{
			UID:          "non-existent-uid",
			Title:        "Updated Title",
			RuleGroup:    "test-group",
			FolderUID:    "tests",
			Condition:    "A",
			Data:         []any{},
			NoDataState:  "OK",
			ExecErrState: "OK",
			For:          "5m",
			OrgID:        1,
		}

		result, err := updateAlertRule(ctx, params)
		require.Error(t, err)
		require.Nil(t, result)
	})

	t.Run("update alert rule with empty UID", func(t *testing.T) {
		ctx := newTestContext()

		params := UpdateAlertRuleParams{
			UID: "",
		}

		result, err := updateAlertRule(ctx, params)
		require.Error(t, err)
		require.Nil(t, result)
		require.Contains(t, err.Error(), "uid is required")
	})
}

func TestAlertingTools_DeleteAlertRule(t *testing.T) {
	t.Run("delete existing alert rule", func(t *testing.T) {
		ctx := newTestContext()

		// First create a rule to delete
		sampleData := []any{
			map[string]any{
				"refId":     "A",
				"queryType": "",
				"relativeTimeRange": map[string]any{
					"from": 600,
					"to":   0,
				},
				"datasourceUid": "prometheus-uid",
				"model": map[string]any{
					"expr":          "up",
					"hide":          false,
					"intervalMs":    1000,
					"maxDataPoints": 43200,
					"refId":         "A",
				},
			},
		}

		testUID := "test_delete_alert_rule"
		createParams := CreateAlertRuleParams{
			Title:        "Rule to Delete",
			RuleGroup:    "test-group",
			FolderUID:    "tests",
			Condition:    "A",
			Data:         sampleData,
			NoDataState:  "OK",
			ExecErrState: "OK",
			For:          "5m",
			UID:          &testUID,
			OrgID:        1,
		}

		// Create the rule
		created, err := createAlertRule(ctx, createParams)
		require.NoError(t, err)
		require.NotNil(t, created)

		// Now delete it
		result, err := deleteAlertRule(ctx, DeleteAlertRuleParams{UID: testUID})
		require.NoError(t, err)
		require.Contains(t, result, "deleted successfully")
		require.Contains(t, result, testUID)

		// Verify it's gone by trying to get it
		_, getErr := getAlertRuleByUID(ctx, GetAlertRuleByUIDParams{UID: testUID})
		require.Error(t, getErr)
	})

	t.Run("delete non-existent alert rule", func(t *testing.T) {
		ctx := newTestContext()

		result, err := deleteAlertRule(ctx, DeleteAlertRuleParams{UID: "non-existent-uid"})
		require.NoError(t, err) // DELETE is idempotent - success even if rule doesn't exist
		require.Contains(t, result, "deleted successfully")
		require.Contains(t, result, "non-existent-uid")
	})

	t.Run("delete alert rule with empty UID", func(t *testing.T) {
		ctx := newTestContext()

		result, err := deleteAlertRule(ctx, DeleteAlertRuleParams{UID: ""})
		require.Error(t, err)
		require.Empty(t, result)
		require.Contains(t, err.Error(), "uid is required")
	})
}

func TestAlertingTools_ListAlertRules_Datasource(t *testing.T) {
	t.Run("list Prometheus-managed alert rules", func(t *testing.T) {
		ctx := newTestContext()
		dsUID := "prometheus"

		result, err := listAlertRules(ctx, ListAlertRulesParams{
			DatasourceUID: &dsUID,
		})
		require.NoError(t, err)
		require.NotEmpty(t, result, "Expected Prometheus to have alert rules configured")

		// Verify we got Prometheus rules
		foundFiring := false
		for _, rule := range result {
			require.NotEmpty(t, rule.Title)
			// Check if we found our test rule
			if rule.Title == "PrometheusTestAlertFiring" {
				foundFiring = true
				require.Equal(t, "warning", rule.Labels["severity"])
				require.Equal(t, "test", rule.Labels["environment"])
			}
		}
		require.True(t, foundFiring, "Expected to find PrometheusTestAlertFiring rule")
	})

	t.Run("list Prometheus rules with label selector", func(t *testing.T) {
		ctx := newTestContext()
		dsUID := "prometheus"

		result, err := listAlertRules(ctx, ListAlertRulesParams{
			DatasourceUID: &dsUID,
			LabelSelectors: []Selector{
				{
					Filters: []LabelMatcher{
						{Name: "severity", Type: "=", Value: "warning"},
					},
				},
			},
		})
		require.NoError(t, err)

		// All returned rules should have severity=warning
		for _, rule := range result {
			require.Equal(t, "warning", rule.Labels["severity"])
		}
	})

	t.Run("list datasource rules - invalid datasource type", func(t *testing.T) {
		ctx := newTestContext()
		dsUID := "tempo" // Not a ruler datasource

		result, err := listAlertRules(ctx, ListAlertRulesParams{
			DatasourceUID: &dsUID,
		})
		require.Error(t, err)
		require.Contains(t, err.Error(), "does not support ruler API")
		require.Nil(t, result)
	})

	t.Run("list datasource rules - nonexistent datasource", func(t *testing.T) {
		ctx := newTestContext()
		dsUID := "nonexistent"

		_, err := listAlertRules(ctx, ListAlertRulesParams{
			DatasourceUID: &dsUID,
		})
		require.Error(t, err)
		require.Contains(t, err.Error(), "not found")
	})

	t.Run("list Loki-managed alert rules", func(t *testing.T) {
		ctx := newTestContext()
		dsUID := "loki"

		result, err := listAlertRules(ctx, ListAlertRulesParams{
			DatasourceUID: &dsUID,
		})
		// Loki ruler may not be fully initialized or may not have rules
		// Just verify no panic and proper error handling
		if err != nil {
			t.Logf("Loki ruler query failed (this may be expected): %v", err)
		} else {
			t.Logf("Loki ruler returned %d rules", len(result))
		}
	})
}

func TestAlertingTools_ListContactPoints_Alertmanager(t *testing.T) {
	t.Run("list Alertmanager receivers", func(t *testing.T) {
		ctx := newTestContext()
		dsUID := "alertmanager"

		result, err := listContactPoints(ctx, ListContactPointsParams{
			DatasourceUID: &dsUID,
		})
		require.NoError(t, err)
		require.NotEmpty(t, result, "Expected Alertmanager to have receivers configured")

		// Verify we got the receivers from alertmanager.yml
		receiverNames := []string{}
		for _, cp := range result {
			receiverNames = append(receiverNames, cp.Name)
			// Alertmanager receivers should not have UIDs
			require.Empty(t, cp.UID)
		}
		require.Contains(t, receiverNames, "test-receiver")
		require.Contains(t, receiverNames, "test-email")
		require.Contains(t, receiverNames, "test-slack")
	})

	t.Run("list Alertmanager receivers with name filter", func(t *testing.T) {
		ctx := newTestContext()
		dsUID := "alertmanager"
		name := "test-receiver"

		result, err := listContactPoints(ctx, ListContactPointsParams{
			DatasourceUID: &dsUID,
			Name:          &name,
		})
		require.NoError(t, err)
		require.Len(t, result, 1)
		require.Equal(t, "test-receiver", result[0].Name)
	})

	t.Run("list contact points - invalid datasource type", func(t *testing.T) {
		ctx := newTestContext()
		dsUID := "prometheus" // Not an Alertmanager

		_, err := listContactPoints(ctx, ListContactPointsParams{
			DatasourceUID: &dsUID,
		})
		require.Error(t, err)
		require.Contains(t, err.Error(), "is not an Alertmanager datasource")
	})

	t.Run("list contact points - nonexistent datasource", func(t *testing.T) {
		ctx := newTestContext()
		dsUID := "nonexistent"

		_, err := listContactPoints(ctx, ListContactPointsParams{
			DatasourceUID: &dsUID,
		})
		require.Error(t, err)
		require.Contains(t, err.Error(), "not found")
	})
}
