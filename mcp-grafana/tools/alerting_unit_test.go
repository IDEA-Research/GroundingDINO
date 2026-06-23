package tools

import (
	"testing"

	"github.com/stretchr/testify/require"
)

// Unit tests for parameter validation (no integration tag needed)
func TestCreateAlertRuleParams_Validate(t *testing.T) {
	t.Run("valid parameters", func(t *testing.T) {
		params := CreateAlertRuleParams{
			Title:        "Test Rule",
			RuleGroup:    "test-group",
			FolderUID:    "test-folder",
			Condition:    "A",
			Data:         []interface{}{map[string]interface{}{"refId": "A"}},
			NoDataState:  "OK",
			ExecErrState: "OK",
			For:          "5m",
			OrgID:        1,
		}
		err := params.validate()
		require.NoError(t, err)
	})

	t.Run("missing title", func(t *testing.T) {
		params := CreateAlertRuleParams{
			RuleGroup:    "test-group",
			FolderUID:    "test-folder",
			Condition:    "A",
			Data:         []interface{}{map[string]interface{}{"refId": "A"}},
			NoDataState:  "OK",
			ExecErrState: "OK",
			For:          "5m",
			OrgID:        1,
		}
		err := params.validate()
		require.Error(t, err)
		require.Contains(t, err.Error(), "title is required")
	})

	t.Run("missing rule group", func(t *testing.T) {
		params := CreateAlertRuleParams{
			Title:        "Test Rule",
			FolderUID:    "test-folder",
			Condition:    "A",
			Data:         []interface{}{map[string]interface{}{"refId": "A"}},
			NoDataState:  "OK",
			ExecErrState: "OK",
			For:          "5m",
			OrgID:        1,
		}
		err := params.validate()
		require.Error(t, err)
		require.Contains(t, err.Error(), "ruleGroup is required")
	})

	t.Run("missing folder UID", func(t *testing.T) {
		params := CreateAlertRuleParams{
			Title:        "Test Rule",
			RuleGroup:    "test-group",
			Condition:    "A",
			Data:         []interface{}{map[string]interface{}{"refId": "A"}},
			NoDataState:  "OK",
			ExecErrState: "OK",
			For:          "5m",
			OrgID:        1,
		}
		err := params.validate()
		require.Error(t, err)
		require.Contains(t, err.Error(), "folderUID is required")
	})

	t.Run("missing condition", func(t *testing.T) {
		params := CreateAlertRuleParams{
			Title:        "Test Rule",
			RuleGroup:    "test-group",
			FolderUID:    "test-folder",
			Data:         []interface{}{map[string]interface{}{"refId": "A"}},
			NoDataState:  "OK",
			ExecErrState: "OK",
			For:          "5m",
			OrgID:        1,
		}
		err := params.validate()
		require.Error(t, err)
		require.Contains(t, err.Error(), "condition is required")
	})

	t.Run("missing data", func(t *testing.T) {
		params := CreateAlertRuleParams{
			Title:        "Test Rule",
			RuleGroup:    "test-group",
			FolderUID:    "test-folder",
			Condition:    "A",
			NoDataState:  "OK",
			ExecErrState: "OK",
			For:          "5m",
			OrgID:        1,
		}
		err := params.validate()
		require.Error(t, err)
		require.Contains(t, err.Error(), "data is required")
	})

	t.Run("missing no data state", func(t *testing.T) {
		params := CreateAlertRuleParams{
			Title:        "Test Rule",
			RuleGroup:    "test-group",
			FolderUID:    "test-folder",
			Condition:    "A",
			Data:         []interface{}{map[string]interface{}{"refId": "A"}},
			ExecErrState: "OK",
			For:          "5m",
			OrgID:        1,
		}
		err := params.validate()
		require.Error(t, err)
		require.Contains(t, err.Error(), "noDataState is required")
	})

	t.Run("missing exec error state", func(t *testing.T) {
		params := CreateAlertRuleParams{
			Title:       "Test Rule",
			RuleGroup:   "test-group",
			FolderUID:   "test-folder",
			Condition:   "A",
			Data:        []interface{}{map[string]interface{}{"refId": "A"}},
			NoDataState: "OK",
			For:         "5m",
			OrgID:       1,
		}
		err := params.validate()
		require.Error(t, err)
		require.Contains(t, err.Error(), "execErrState is required")
	})

	t.Run("missing for duration", func(t *testing.T) {
		params := CreateAlertRuleParams{
			Title:        "Test Rule",
			RuleGroup:    "test-group",
			FolderUID:    "test-folder",
			Condition:    "A",
			Data:         []interface{}{map[string]interface{}{"refId": "A"}},
			NoDataState:  "OK",
			ExecErrState: "OK",
			OrgID:        1,
		}
		err := params.validate()
		require.Error(t, err)
		require.Contains(t, err.Error(), "for duration is required")
	})

	t.Run("invalid org ID", func(t *testing.T) {
		params := CreateAlertRuleParams{
			Title:        "Test Rule",
			RuleGroup:    "test-group",
			FolderUID:    "test-folder",
			Condition:    "A",
			Data:         []interface{}{map[string]interface{}{"refId": "A"}},
			NoDataState:  "OK",
			ExecErrState: "OK",
			For:          "5m",
			OrgID:        0,
		}
		err := params.validate()
		require.Error(t, err)
		require.Contains(t, err.Error(), "orgID is required and must be greater than 0")
	})
}

func TestUpdateAlertRuleParams_Validate(t *testing.T) {
	t.Run("valid parameters", func(t *testing.T) {
		params := UpdateAlertRuleParams{
			UID:          "test-uid",
			Title:        "Test Rule",
			RuleGroup:    "test-group",
			FolderUID:    "test-folder",
			Condition:    "A",
			Data:         []interface{}{map[string]interface{}{"refId": "A"}},
			NoDataState:  "OK",
			ExecErrState: "OK",
			For:          "5m",
			OrgID:        1,
		}
		err := params.validate()
		require.NoError(t, err)
	})

	t.Run("missing UID", func(t *testing.T) {
		params := UpdateAlertRuleParams{
			Title:        "Test Rule",
			RuleGroup:    "test-group",
			FolderUID:    "test-folder",
			Condition:    "A",
			Data:         []interface{}{map[string]interface{}{"refId": "A"}},
			NoDataState:  "OK",
			ExecErrState: "OK",
			For:          "5m",
			OrgID:        1,
		}
		err := params.validate()
		require.Error(t, err)
		require.Contains(t, err.Error(), "uid is required")
	})

	t.Run("invalid org ID", func(t *testing.T) {
		params := UpdateAlertRuleParams{
			UID:          "test-uid",
			Title:        "Test Rule",
			RuleGroup:    "test-group",
			FolderUID:    "test-folder",
			Condition:    "A",
			Data:         []interface{}{map[string]interface{}{"refId": "A"}},
			NoDataState:  "OK",
			ExecErrState: "OK",
			For:          "5m",
			OrgID:        -1,
		}
		err := params.validate()
		require.Error(t, err)
		require.Contains(t, err.Error(), "orgID is required and must be greater than 0")
	})
}

func TestDeleteAlertRuleParams_Validate(t *testing.T) {
	t.Run("valid parameters", func(t *testing.T) {
		params := DeleteAlertRuleParams{
			UID: "test-uid",
		}
		err := params.validate()
		require.NoError(t, err)
	})

	t.Run("missing UID", func(t *testing.T) {
		params := DeleteAlertRuleParams{
			UID: "",
		}
		err := params.validate()
		require.Error(t, err)
		require.Contains(t, err.Error(), "uid is required")
	})
}

func TestBuiltInValidationCatchesInvalidData(t *testing.T) {
	t.Run("invalid NoDataState enum value", func(t *testing.T) {
		params := CreateAlertRuleParams{
			Title:        "Test Rule",
			RuleGroup:    "test-group",
			FolderUID:    "test-folder",
			Condition:    "A",
			Data:         []interface{}{map[string]interface{}{"refId": "A"}},
			NoDataState:  "InvalidValue", // Invalid enum
			ExecErrState: "OK",
			For:          "5m",
			OrgID:        1,
		}

		// Our simple validation won't catch this, but it would fail at API call
		err := params.validate()
		require.NoError(t, err, "Simple validation doesn't check enum values")
	})

	t.Run("invalid ExecErrState enum value", func(t *testing.T) {
		params := CreateAlertRuleParams{
			Title:        "Test Rule",
			RuleGroup:    "test-group",
			FolderUID:    "test-folder",
			Condition:    "A",
			Data:         []interface{}{map[string]interface{}{"refId": "A"}},
			NoDataState:  "OK",
			ExecErrState: "BadValue", // Invalid enum
			For:          "5m",
			OrgID:        1,
		}

		// Our simple validation won't catch this
		err := params.validate()
		require.NoError(t, err, "Simple validation doesn't check enum values")
	})

	t.Run("title too long", func(t *testing.T) {
		longTitle := make([]byte, 200) // Max is 190
		for i := range longTitle {
			longTitle[i] = 'A'
		}

		params := CreateAlertRuleParams{
			Title:        string(longTitle),
			RuleGroup:    "test-group",
			FolderUID:    "test-folder",
			Condition:    "A",
			Data:         []interface{}{map[string]interface{}{"refId": "A"}},
			NoDataState:  "OK",
			ExecErrState: "OK",
			For:          "5m",
			OrgID:        1,
		}

		// Simple validation only checks if title is empty, not length
		err := params.validate()
		require.NoError(t, err, "Simple validation doesn't check length constraints")
	})
}
