// Requires a Grafana instance running on localhost:3000,
// with a dashboard provisioned.
// Run with `go test -tags integration`.
//go:build integration

package tools

import (
	"context"
	"testing"

	"github.com/grafana/grafana-openapi-client-go/models"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

const (
	newTestDashboardName = "Integration Test"
)

// getExistingDashboardUID will fetch an existing dashboard for test purposes
// It will search for exisiting dashboards and return the first, otherwise
// will trigger a test error
func getExistingTestDashboard(t *testing.T, ctx context.Context, dashboardName string) *models.Hit {
	// Make sure we query for the existing dashboard, not a folder
	if dashboardName == "" {
		dashboardName = "Demo"
	}
	searchResults, err := searchDashboards(ctx, SearchDashboardsParams{
		Query: dashboardName,
	})
	require.NoError(t, err)
	require.Greater(t, len(searchResults), 0, "No dashboards found")
	return searchResults[0]
}

// getExistingTestDashboardJSON will fetch the JSON map for an existing
// dashboard in the test environment
func getTestDashboardJSON(t *testing.T, ctx context.Context, dashboard *models.Hit) map[string]interface{} {
	result, err := getDashboardByUID(ctx, GetDashboardByUIDParams{
		UID: dashboard.UID,
	})
	require.NoError(t, err)
	dashboardMap, ok := result.Dashboard.(map[string]interface{})
	require.True(t, ok, "Dashboard should be a map")
	return dashboardMap
}

func TestDashboardTools(t *testing.T) {
	t.Run("get dashboard by uid", func(t *testing.T) {
		ctx := newTestContext()

		// First, let's search for a dashboard to get its UID
		dashboard := getExistingTestDashboard(t, ctx, "")

		// Now test the get dashboard by uid functionality
		result, err := getDashboardByUID(ctx, GetDashboardByUIDParams{
			UID: dashboard.UID,
		})
		require.NoError(t, err)
		dashboardMap, ok := result.Dashboard.(map[string]interface{})
		require.True(t, ok, "Dashboard should be a map")
		assert.Equal(t, dashboard.UID, dashboardMap["uid"])
		assert.NotNil(t, result.Meta)
	})

	t.Run("get dashboard by uid - invalid uid", func(t *testing.T) {
		ctx := newTestContext()

		_, err := getDashboardByUID(ctx, GetDashboardByUIDParams{
			UID: "non-existent-uid",
		})
		require.Error(t, err)
	})

	t.Run("update dashboard - create new", func(t *testing.T) {
		ctx := newTestContext()

		// Get the dashboard JSON
		// In this case, we will create a new dashboard with the same
		// content but different Title, and disable "overwrite"
		dashboard := getExistingTestDashboard(t, ctx, "")
		dashboardMap := getTestDashboardJSON(t, ctx, dashboard)

		// Avoid a clash by unsetting the existing IDs
		delete(dashboardMap, "uid")
		delete(dashboardMap, "id")

		// Set a new title and tag
		dashboardMap["title"] = newTestDashboardName
		dashboardMap["tags"] = []string{"integration-test"}

		params := UpdateDashboardParams{
			Dashboard: dashboardMap,
			Message:   "creating a new dashboard",
			Overwrite: false,
			UserID:    1,
		}

		// Only pass in the Folder UID if it exists
		if dashboard.FolderUID != "" {
			params.FolderUID = dashboard.FolderUID
		}

		// create the dashboard
		_, err := updateDashboard(ctx, params)
		require.NoError(t, err)
	})

	t.Run("update dashboard - overwrite existing", func(t *testing.T) {
		ctx := newTestContext()

		// Get the dashboard JSON for the non-provisioned dashboard we've created
		dashboard := getExistingTestDashboard(t, ctx, newTestDashboardName)
		dashboardMap := getTestDashboardJSON(t, ctx, dashboard)

		params := UpdateDashboardParams{
			Dashboard: dashboardMap,
			Message:   "updating existing dashboard",
			Overwrite: true,
			UserID:    1,
		}

		// Only pass in the Folder UID if it exists
		if dashboard.FolderUID != "" {
			params.FolderUID = dashboard.FolderUID
		}

		// update the dashboard
		_, err := updateDashboard(ctx, params)
		require.NoError(t, err)
	})

	t.Run("get dashboard panel queries", func(t *testing.T) {
		ctx := newTestContext()

		// Get the test dashboard
		dashboard := getExistingTestDashboard(t, ctx, "")

		result, err := GetDashboardPanelQueriesTool(ctx, DashboardPanelQueriesParams{
			UID: dashboard.UID,
		})
		require.NoError(t, err)
		assert.Greater(t, len(result), 0, "Should return at least one panel query")

		// The initial demo dashboard plus for all dashboards created by the integration tests,
		// every panel should have identical title and query values.
		// Datasource UID may differ. Datasource type can be an empty string as well but on the demo and test dashboards it should be "prometheus".
		for _, panelQuery := range result {
			assert.Equal(t, panelQuery.Title, "Node Load")
			assert.Equal(t, panelQuery.Query, "node_load1")
			assert.NotEmpty(t, panelQuery.Datasource.UID)
			assert.Equal(t, panelQuery.Datasource.Type, "prometheus")
		}
	})

	// Tests for new Issue #101 context window management tools
	t.Run("get dashboard summary", func(t *testing.T) {
		ctx := newTestContext()

		// Get the test dashboard
		dashboard := getExistingTestDashboard(t, ctx, "")

		result, err := getDashboardSummary(ctx, GetDashboardSummaryParams{
			UID: dashboard.UID,
		})
		require.NoError(t, err)

		assert.Equal(t, dashboard.UID, result.UID)
		assert.NotEmpty(t, result.Title)
		assert.Greater(t, result.PanelCount, 0, "Should have at least one panel")
		assert.Len(t, result.Panels, result.PanelCount, "Panel count should match panels array length")
		assert.NotNil(t, result.Meta)

		// Check that panels have expected structure
		for _, panel := range result.Panels {
			assert.NotEmpty(t, panel.Title)
			assert.NotEmpty(t, panel.Type)
			assert.GreaterOrEqual(t, panel.QueryCount, 0)
		}
	})

	t.Run("get dashboard property - title", func(t *testing.T) {
		ctx := newTestContext()

		dashboard := getExistingTestDashboard(t, ctx, "")

		result, err := getDashboardProperty(ctx, GetDashboardPropertyParams{
			UID:      dashboard.UID,
			JSONPath: "$.title",
		})
		require.NoError(t, err)

		title, ok := result.(string)
		require.True(t, ok, "Title should be a string")
		assert.NotEmpty(t, title)
	})

	t.Run("get dashboard property - panel titles", func(t *testing.T) {
		ctx := newTestContext()

		dashboard := getExistingTestDashboard(t, ctx, "")

		result, err := getDashboardProperty(ctx, GetDashboardPropertyParams{
			UID:      dashboard.UID,
			JSONPath: "$.panels[*].title",
		})
		require.NoError(t, err)

		titles, ok := result.([]interface{})
		require.True(t, ok, "Panel titles should be an array")
		assert.Greater(t, len(titles), 0, "Should have at least one panel title")

		for _, title := range titles {
			titleStr, ok := title.(string)
			require.True(t, ok, "Each title should be a string")
			assert.NotEmpty(t, titleStr)
		}
	})

	t.Run("get dashboard property - invalid path", func(t *testing.T) {
		ctx := newTestContext()

		dashboard := getExistingTestDashboard(t, ctx, "")

		_, err := getDashboardProperty(ctx, GetDashboardPropertyParams{
			UID:      dashboard.UID,
			JSONPath: "$.nonexistent.path",
		})
		require.Error(t, err, "Should fail for non-existent path")
	})

	t.Run("update dashboard - patch title", func(t *testing.T) {
		ctx := newTestContext()

		// Get our test dashboard (not the provisioned one)
		dashboard := getExistingTestDashboard(t, ctx, newTestDashboardName)

		newTitle := "Updated Integration Test Dashboard"

		result, err := updateDashboard(ctx, UpdateDashboardParams{
			UID: dashboard.UID,
			Operations: []PatchOperation{
				{
					Op:    "replace",
					Path:  "$.title",
					Value: newTitle,
				},
			},
			Message: "Updated title via patch",
		})
		require.NoError(t, err)
		assert.NotNil(t, result)

		// Verify the change was applied
		updatedDashboard, err := getDashboardByUID(ctx, GetDashboardByUIDParams{
			UID: dashboard.UID,
		})
		require.NoError(t, err)

		dashboardMap, ok := updatedDashboard.Dashboard.(map[string]interface{})
		require.True(t, ok, "Dashboard should be a map")
		assert.Equal(t, newTitle, dashboardMap["title"])
	})

	t.Run("update dashboard - patch add description", func(t *testing.T) {
		ctx := newTestContext()

		dashboard := getExistingTestDashboard(t, ctx, newTestDashboardName)

		description := "This is a test description added via patch"

		_, err := updateDashboard(ctx, UpdateDashboardParams{
			UID: dashboard.UID,
			Operations: []PatchOperation{
				{
					Op:    "add",
					Path:  "$.description",
					Value: description,
				},
			},
			Message: "Added description via patch",
		})
		require.NoError(t, err)

		// Verify the description was added
		updatedDashboard, err := getDashboardByUID(ctx, GetDashboardByUIDParams{
			UID: dashboard.UID,
		})
		require.NoError(t, err)

		dashboardMap, ok := updatedDashboard.Dashboard.(map[string]interface{})
		require.True(t, ok, "Dashboard should be a map")
		assert.Equal(t, description, dashboardMap["description"])
	})

	t.Run("update dashboard - patch remove description", func(t *testing.T) {
		ctx := newTestContext()

		dashboard := getExistingTestDashboard(t, ctx, newTestDashboardName)

		_, err := updateDashboard(ctx, UpdateDashboardParams{
			UID: dashboard.UID,
			Operations: []PatchOperation{
				{
					Op:   "remove",
					Path: "$.description",
				},
			},
			Message: "Removed description via patch",
		})
		require.NoError(t, err)

		// Verify the description was removed
		updatedDashboard, err := getDashboardByUID(ctx, GetDashboardByUIDParams{
			UID: dashboard.UID,
		})
		require.NoError(t, err)

		dashboardMap, ok := updatedDashboard.Dashboard.(map[string]interface{})
		require.True(t, ok, "Dashboard should be a map")
		_, hasDescription := dashboardMap["description"]
		assert.False(t, hasDescription, "Description should be removed")
	})

	t.Run("update dashboard - unsupported operation", func(t *testing.T) {
		ctx := newTestContext()

		dashboard := getExistingTestDashboard(t, ctx, newTestDashboardName)

		_, err := updateDashboard(ctx, UpdateDashboardParams{
			UID: dashboard.UID,
			Operations: []PatchOperation{
				{
					Op:    "copy", // Unsupported operation
					Path:  "$.title",
					Value: "New Title",
				},
			},
		})
		require.Error(t, err, "Should fail for unsupported operation")
	})

	t.Run("update dashboard - invalid parameters", func(t *testing.T) {
		ctx := newTestContext()

		_, err := updateDashboard(ctx, UpdateDashboardParams{
			// Neither dashboard nor (uid + operations) provided
		})
		require.Error(t, err, "Should fail when no valid parameters provided")
	})

	t.Run("update dashboard - append to panels array", func(t *testing.T) {
		ctx := newTestContext()

		// Get our test dashboard
		dashboard := getExistingTestDashboard(t, ctx, newTestDashboardName)

		// Create a new panel to append
		newPanel := map[string]interface{}{
			"id":    999,
			"title": "New Appended Panel",
			"type":  "stat",
			"targets": []interface{}{
				map[string]interface{}{
					"expr": "up",
				},
			},
			"gridPos": map[string]interface{}{
				"h": 8,
				"w": 12,
				"x": 0,
				"y": 8,
			},
		}

		_, err := updateDashboard(ctx, UpdateDashboardParams{
			UID: dashboard.UID,
			Operations: []PatchOperation{
				{
					Op:    "add",
					Path:  "$.panels/-",
					Value: newPanel,
				},
			},
			Message: "Appended new panel via /- syntax",
		})
		require.NoError(t, err)

		// Verify the panel was appended
		updatedDashboard, err := getDashboardByUID(ctx, GetDashboardByUIDParams{
			UID: dashboard.UID,
		})
		require.NoError(t, err)

		dashboardMap, ok := updatedDashboard.Dashboard.(map[string]interface{})
		require.True(t, ok, "Dashboard should be a map")

		panels, ok := dashboardMap["panels"].([]interface{})
		require.True(t, ok, "Panels should be an array")

		// Check that the new panel was appended (should be the last panel)
		lastPanel, ok := panels[len(panels)-1].(map[string]interface{})
		require.True(t, ok, "Last panel should be an object")
		assert.Equal(t, "New Appended Panel", lastPanel["title"])
		assert.Equal(t, float64(999), lastPanel["id"]) // JSON unmarshaling converts to float64
	})

	t.Run("update dashboard - remove with append syntax should fail", func(t *testing.T) {
		ctx := newTestContext()

		dashboard := getExistingTestDashboard(t, ctx, newTestDashboardName)

		_, err := updateDashboard(ctx, UpdateDashboardParams{
			UID: dashboard.UID,
			Operations: []PatchOperation{
				{
					Op:   "remove",
					Path: "$.panels/-", // Invalid: remove with append syntax
				},
			},
		})
		require.Error(t, err, "Should fail when using remove operation with append syntax")
	})

	t.Run("update dashboard - append to non-array should fail", func(t *testing.T) {
		ctx := newTestContext()

		dashboard := getExistingTestDashboard(t, ctx, newTestDashboardName)

		_, err := updateDashboard(ctx, UpdateDashboardParams{
			UID: dashboard.UID,
			Operations: []PatchOperation{
				{
					Op:    "add",
					Path:  "$.title/-", // Invalid: title is not an array
					Value: "Invalid",
				},
			},
		})
		require.Error(t, err, "Should fail when trying to append to non-array field")
	})
}
