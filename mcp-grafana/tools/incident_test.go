//go:build unit
// +build unit

package tools

import (
	"context"
	"testing"

	"github.com/grafana/incident-go"
	mcpgrafana "github.com/grafana/mcp-grafana"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func newIncidentTestContext() context.Context {
	client := incident.NewTestClient()
	return mcpgrafana.WithIncidentClient(context.Background(), client)
}

func TestIncidentTools(t *testing.T) {
	t.Run("list incidents", func(t *testing.T) {
		ctx := newIncidentTestContext()
		result, err := listIncidents(ctx, ListIncidentsParams{
			Limit: 2,
		})
		require.NoError(t, err)
		assert.Len(t, result.IncidentPreviews, 2)
	})

	t.Run("create incident", func(t *testing.T) {
		ctx := newIncidentTestContext()
		result, err := createIncident(ctx, CreateIncidentParams{
			Title:         "high latency in web requests",
			Severity:      "minor",
			RoomPrefix:    "test",
			IsDrill:       true,
			Status:        "active",
			AttachCaption: "Test attachment",
			AttachURL:     "https://grafana.com",
		})
		require.NoError(t, err)
		assert.Equal(t, "high latency in web requests", result.Title)
		assert.Equal(t, "minor", result.Severity)
		assert.True(t, result.IsDrill)
		assert.Equal(t, "active", result.Status)
	})

	t.Run("add activity to incident", func(t *testing.T) {
		ctx := newIncidentTestContext()
		result, err := addActivityToIncident(ctx, AddActivityToIncidentParams{
			IncidentID: "123",
			Body:       "The incident was created by user-123",
			EventTime:  "2021-08-07T11:58:23Z",
		})
		require.NoError(t, err)
		assert.Equal(t, "The incident was created by user-123", result.Body)
		assert.Equal(t, "2021-08-07T11:58:23Z", result.EventTime)
	})
}
