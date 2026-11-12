/*
 * GreenFed-XAI Smart Contract — Hyperledger Fabric 2.4+ Chaincode
 * ================================================================
 *
 * This Go chaincode implements the on-chain verification protocol
 * for the blockchain-verified federated learning system described
 * in the IEEE paper: "Blockchain-Verified Federated Learning for
 * ESG-Driven Green Bond Pricing."
 *
 * Smart Contract Functions:
 *
 *   SubmitHash(clientID, roundIdx, parameterHash):
 *       Records the SHA-256 hash of a client's model update on the
 *       Hyperledger Fabric world state. This establishes a tamper-proof
 *       cryptographic commitment before the aggregation step.
 *
 *   VerifyUpdate(clientID, roundIdx, computedHash, gradientNorm):
 *       Performs two-phase verification:
 *         Phase 1: Checks that the computed hash matches the previously
 *                  submitted hash (integrity check).
 *         Phase 2: Checks that the gradient norm ||Δw||_2 ≤ τ (2.5)
 *                  (poisoning detection).
 *       Returns the verification status (ACCEPTED / REJECTED_*).
 *
 *   RejectPoisoned(clientID, roundIdx, reason):
 *       Records a permanent rejection event on the ledger. This creates
 *       an immutable audit trail of all poisoned update attempts.
 *
 *   QueryHash(clientID, roundIdx):
 *       Retrieves the stored parameter hash for a given client and round
 *       from the world state.
 *
 *   QueryAuditTrail(roundIdx):
 *       Returns the complete verification history for a given FL round,
 *       including all submissions, acceptances, and rejections.
 *
 * Deployment:
 *   This chaincode is designed for Hyperledger Fabric test-network
 *   deployment using the peer CLI:
 *
 *     1. Package: peer lifecycle chaincode package greenfed.tar.gz ...
 *     2. Install:  peer lifecycle chaincode install greenfed.tar.gz
 *     3. Approve:  peer lifecycle chaincode approveformyorg ...
 *     4. Commit:   peer lifecycle chaincode commit ...
 *     5. Invoke:   peer chaincode invoke -C mychannel -n greenfed \
 *                    -c '{"function":"SubmitHash","Args":["bank0","1","abc123"]}'
 *
 * Note:
 *   For local development and unit testing, the Python mock_ledger.py
 *   provides equivalent functionality without requiring Docker or
 *   Fabric infrastructure. This Go chaincode is the production-ready
 *   Hyperledger Fabric implementation.
 *
 * Author:  GreenFed-XAI Project
 * License: Apache 2.0
 * Paper:   "Blockchain-Verified FL for ESG-Driven Green Bond Pricing"
 */

package main

import (
	"encoding/json"
	"fmt"
	"math"
	"strconv"
	"time"

	"github.com/hyperledger/fabric-contract-api-go/contractapi"
)

// =========================================================================
// Constants — Paper-Specified Parameters
// =========================================================================

const (
	// GradientNormThreshold is the τ value from the paper (Section IV-D).
	// Updates with ||Δw||_2 > τ are classified as potentially poisoned.
	GradientNormThreshold float64 = 2.5

	// HashAlgorithm identifies the cryptographic hash function used.
	HashAlgorithm string = "SHA-256"

	// HashBytesStored is the size of a SHA-256 hash in bytes.
	// The paper notes only 32 bytes are stored per round per client.
	HashBytesStored int = 32
)

// =========================================================================
// Data Structures
// =========================================================================

// HashSubmission represents a client's committed parameter hash stored
// on the Hyperledger Fabric world state.
type HashSubmission struct {
	ClientID      string  `json:"client_id"`
	RoundIdx      int     `json:"round_idx"`
	ParameterHash string  `json:"parameter_hash"`
	HashAlgorithm string  `json:"hash_algorithm"`
	BytesStored   int     `json:"bytes_stored"`
	SubmittedAt   string  `json:"submitted_at"`
	Status        string  `json:"status"` // PENDING, VERIFIED, ACCEPTED, REJECTED
}

// VerificationRecord captures the result of verifying a client's model
// update against the committed hash and gradient norm threshold.
type VerificationRecord struct {
	ClientID           string  `json:"client_id"`
	RoundIdx           int     `json:"round_idx"`
	ComputedHash       string  `json:"computed_hash"`
	SubmittedHash      string  `json:"submitted_hash"`
	HashValid          bool    `json:"hash_valid"`
	GradientNorm       float64 `json:"gradient_norm"`
	NormThreshold      float64 `json:"norm_threshold"`
	NormValid          bool    `json:"norm_valid"`
	Status             string  `json:"status"` // ACCEPTED, REJECTED_HASH, REJECTED_NORM, REJECTED_BOTH
	Reason             string  `json:"reason"`
	VerifiedAt         string  `json:"verified_at"`
	VerificationTimeMs float64 `json:"verification_time_ms"`
}

// RoundAuditTrail contains the complete verification history for one
// federated learning communication round.
type RoundAuditTrail struct {
	RoundIdx         int                    `json:"round_idx"`
	TotalSubmissions int                    `json:"total_submissions"`
	TotalAccepted    int                    `json:"total_accepted"`
	TotalRejected    int                    `json:"total_rejected"`
	AcceptanceRate   float64                `json:"acceptance_rate"`
	Records          []VerificationRecord   `json:"records"`
	GeneratedAt      string                 `json:"generated_at"`
}

// =========================================================================
// Smart Contract Definition
// =========================================================================

// GreenFedContract implements the chaincode interface for the
// blockchain-verified federated learning verification protocol.
type GreenFedContract struct {
	contractapi.Contract
}

// =========================================================================
// SubmitHash — Record Parameter Hash Commitment
// =========================================================================

// SubmitHash stores the SHA-256 hash of a client's model update
// parameters on the Hyperledger Fabric world state.
//
// This function is called by each participating bank after completing
// local training and computing the SHA-256 hash of their updated model
// parameters. The hash serves as a tamper-proof commitment that is
// verified before the server aggregates the update.
//
// World State Key Format: "HASH_{clientID}_{roundIdx}"
//
// Parameters:
//   - ctx: Transaction context provided by the Fabric runtime.
//   - clientID: Unique identifier of the submitting bank (e.g., "bank_0").
//   - roundIdxStr: FL communication round index as a string.
//   - parameterHash: SHA-256 hex digest of the model parameters (64 chars).
//
// Returns:
//   - error: nil on success, or an error if the submission fails.
func (c *GreenFedContract) SubmitHash(
	ctx contractapi.TransactionContextInterface,
	clientID string,
	roundIdxStr string,
	parameterHash string,
) error {
	// Parse and validate the round index
	roundIdx, err := strconv.Atoi(roundIdxStr)
	if err != nil {
		return fmt.Errorf("invalid round index '%s': %v", roundIdxStr, err)
	}
	if roundIdx < 0 {
		return fmt.Errorf("round index must be non-negative, got %d", roundIdx)
	}

	// Validate the parameter hash format (64 hex characters for SHA-256)
	if len(parameterHash) != 64 {
		return fmt.Errorf(
			"invalid parameter hash length: expected 64 hex characters, got %d",
			len(parameterHash),
		)
	}

	// Check for duplicate submissions (idempotency guard)
	compositeKey := fmt.Sprintf("HASH_%s_%d", clientID, roundIdx)
	existingBytes, err := ctx.GetStub().GetState(compositeKey)
	if err != nil {
		return fmt.Errorf("failed to read world state: %v", err)
	}
	if existingBytes != nil {
		return fmt.Errorf(
			"hash already submitted for client '%s' in round %d",
			clientID, roundIdx,
		)
	}

	// Create the hash submission record
	submission := HashSubmission{
		ClientID:      clientID,
		RoundIdx:      roundIdx,
		ParameterHash: parameterHash,
		HashAlgorithm: HashAlgorithm,
		BytesStored:   HashBytesStored,
		SubmittedAt:   time.Now().UTC().Format(time.RFC3339),
		Status:        "PENDING",
	}

	// Serialize and store on the world state
	submissionJSON, err := json.Marshal(submission)
	if err != nil {
		return fmt.Errorf("failed to marshal submission: %v", err)
	}

	err = ctx.GetStub().PutState(compositeKey, submissionJSON)
	if err != nil {
		return fmt.Errorf("failed to write to world state: %v", err)
	}

	// Emit a chaincode event for external listeners
	eventPayload, _ := json.Marshal(map[string]interface{}{
		"event":     "HASH_SUBMITTED",
		"client_id": clientID,
		"round_idx": roundIdx,
		"hash":      parameterHash[:16] + "...",
	})
	ctx.GetStub().SetEvent("HashSubmitted", eventPayload)

	return nil
}

// =========================================================================
// VerifyUpdate — Two-Phase Verification Protocol
// =========================================================================

// VerifyUpdate performs the complete two-phase verification of a client's
// model update as specified in Section IV-D of the IEEE paper.
//
// Phase 1 — Hash Integrity Check:
//   The function retrieves the previously committed hash from the world
//   state and compares it with the hash recomputed from the actual
//   submitted parameters. A mismatch indicates potential tampering.
//
// Phase 2 — Gradient Norm Check:
//   The gradient norm ||Δw||_2 is compared against the threshold
//   τ = 2.5. Updates exceeding this threshold exhibit the characteristic
//   signature of model poisoning attacks and are rejected.
//
// Parameters:
//   - ctx: Transaction context.
//   - clientID: Bank identifier.
//   - roundIdxStr: FL round index (string).
//   - computedHash: SHA-256 hash recomputed from actual parameters.
//   - gradientNormStr: L2 norm of the parameter update vector (string).
//
// Returns:
//   - string: JSON-encoded VerificationRecord with the outcome.
//   - error: nil on success, or an error if verification fails.
func (c *GreenFedContract) VerifyUpdate(
	ctx contractapi.TransactionContextInterface,
	clientID string,
	roundIdxStr string,
	computedHash string,
	gradientNormStr string,
) (string, error) {
	startTime := time.Now()

	// Parse round index
	roundIdx, err := strconv.Atoi(roundIdxStr)
	if err != nil {
		return "", fmt.Errorf("invalid round index '%s': %v", roundIdxStr, err)
	}

	// Parse gradient norm
	gradientNorm, err := strconv.ParseFloat(gradientNormStr, 64)
	if err != nil {
		return "", fmt.Errorf("invalid gradient norm '%s': %v", gradientNormStr, err)
	}

	// Retrieve the submitted hash from world state
	compositeKey := fmt.Sprintf("HASH_%s_%d", clientID, roundIdx)
	submissionBytes, err := ctx.GetStub().GetState(compositeKey)
	if err != nil {
		return "", fmt.Errorf("failed to read world state: %v", err)
	}
	if submissionBytes == nil {
		return "", fmt.Errorf(
			"no hash submission found for client '%s' in round %d",
			clientID, roundIdx,
		)
	}

	// Deserialize the stored submission
	var submission HashSubmission
	err = json.Unmarshal(submissionBytes, &submission)
	if err != nil {
		return "", fmt.Errorf("failed to unmarshal submission: %v", err)
	}

	// ---------------------------------------------------------------
	// Phase 1: Hash Integrity Check
	// ---------------------------------------------------------------
	hashValid := computedHash == submission.ParameterHash

	// ---------------------------------------------------------------
	// Phase 2: Gradient Norm Check (||Δw||_2 ≤ τ)
	// ---------------------------------------------------------------
	normValid := gradientNorm <= GradientNormThreshold

	// ---------------------------------------------------------------
	// Determine Verification Status
	// ---------------------------------------------------------------
	var status string
	var reason string

	if hashValid && normValid {
		status = "ACCEPTED"
		reason = "Both hash integrity and gradient norm checks passed"
	} else if !hashValid && !normValid {
		status = "REJECTED_BOTH"
		reason = fmt.Sprintf(
			"Hash mismatch (computed=%s vs submitted=%s) AND "+
				"gradient norm exceeded (||Δw||_2=%.4f > τ=%.1f)",
			computedHash[:16]+"...",
			submission.ParameterHash[:16]+"...",
			gradientNorm,
			GradientNormThreshold,
		)
	} else if !hashValid {
		status = "REJECTED_HASH"
		reason = fmt.Sprintf(
			"Hash mismatch: computed=%s vs submitted=%s",
			computedHash[:16]+"...",
			submission.ParameterHash[:16]+"...",
		)
	} else {
		status = "REJECTED_NORM"
		reason = fmt.Sprintf(
			"Gradient norm exceeded: ||Δw||_2=%.4f > τ=%.1f",
			gradientNorm,
			GradientNormThreshold,
		)
	}

	verificationTimeMs := float64(time.Since(startTime).Microseconds()) / 1000.0

	// Create the verification record
	record := VerificationRecord{
		ClientID:           clientID,
		RoundIdx:           roundIdx,
		ComputedHash:       computedHash,
		SubmittedHash:      submission.ParameterHash,
		HashValid:          hashValid,
		GradientNorm:       math.Round(gradientNorm*10000) / 10000,
		NormThreshold:      GradientNormThreshold,
		NormValid:          normValid,
		Status:             status,
		Reason:             reason,
		VerifiedAt:         time.Now().UTC().Format(time.RFC3339),
		VerificationTimeMs: math.Round(verificationTimeMs*1000) / 1000,
	}

	// Update the submission status on the world state
	submission.Status = status
	updatedSubmissionJSON, _ := json.Marshal(submission)
	ctx.GetStub().PutState(compositeKey, updatedSubmissionJSON)

	// Store the verification record
	verifyKey := fmt.Sprintf("VERIFY_%s_%d", clientID, roundIdx)
	recordJSON, err := json.Marshal(record)
	if err != nil {
		return "", fmt.Errorf("failed to marshal verification record: %v", err)
	}
	ctx.GetStub().PutState(verifyKey, recordJSON)

	// Emit appropriate chaincode event
	eventName := "UpdateVerified"
	if status != "ACCEPTED" {
		eventName = "UpdateRejected"
	}
	eventPayload, _ := json.Marshal(map[string]interface{}{
		"event":     eventName,
		"client_id": clientID,
		"round_idx": roundIdx,
		"status":    status,
	})
	ctx.GetStub().SetEvent(eventName, eventPayload)

	return string(recordJSON), nil
}

// =========================================================================
// RejectPoisoned — Record Rejection Event
// =========================================================================

// RejectPoisoned creates an immutable rejection record on the blockchain
// ledger for audit trail purposes. This function is called when a model
// update fails the verification checks and must be permanently excluded
// from the FedAvg aggregation.
//
// Parameters:
//   - ctx: Transaction context.
//   - clientID: Bank identifier of the rejected client.
//   - roundIdxStr: FL round index (string).
//   - reason: Human-readable rejection reason.
//
// Returns:
//   - error: nil on success, or an error if recording fails.
func (c *GreenFedContract) RejectPoisoned(
	ctx contractapi.TransactionContextInterface,
	clientID string,
	roundIdxStr string,
	reason string,
) error {
	roundIdx, err := strconv.Atoi(roundIdxStr)
	if err != nil {
		return fmt.Errorf("invalid round index '%s': %v", roundIdxStr, err)
	}

	// Create the rejection record
	rejection := map[string]interface{}{
		"client_id":   clientID,
		"round_idx":   roundIdx,
		"reason":      reason,
		"rejected_at": time.Now().UTC().Format(time.RFC3339),
		"tx_type":     "REJECT_POISONED",
	}

	rejectionJSON, err := json.Marshal(rejection)
	if err != nil {
		return fmt.Errorf("failed to marshal rejection: %v", err)
	}

	// Store on world state with a unique key
	rejectKey := fmt.Sprintf("REJECT_%s_%d", clientID, roundIdx)
	err = ctx.GetStub().PutState(rejectKey, rejectionJSON)
	if err != nil {
		return fmt.Errorf("failed to write rejection to world state: %v", err)
	}

	// Emit rejection event
	ctx.GetStub().SetEvent("PoisonedRejected", rejectionJSON)

	return nil
}

// =========================================================================
// QueryHash — Retrieve Stored Parameter Hash
// =========================================================================

// QueryHash retrieves the parameter hash previously submitted by a client
// for a specific FL round from the world state.
//
// Parameters:
//   - ctx: Transaction context.
//   - clientID: Bank identifier.
//   - roundIdxStr: FL round index (string).
//
// Returns:
//   - string: JSON-encoded HashSubmission record.
//   - error: nil on success, or an error if the hash is not found.
func (c *GreenFedContract) QueryHash(
	ctx contractapi.TransactionContextInterface,
	clientID string,
	roundIdxStr string,
) (string, error) {
	roundIdx, err := strconv.Atoi(roundIdxStr)
	if err != nil {
		return "", fmt.Errorf("invalid round index '%s': %v", roundIdxStr, err)
	}

	compositeKey := fmt.Sprintf("HASH_%s_%d", clientID, roundIdx)
	submissionBytes, err := ctx.GetStub().GetState(compositeKey)
	if err != nil {
		return "", fmt.Errorf("failed to read world state: %v", err)
	}
	if submissionBytes == nil {
		return "", fmt.Errorf(
			"no hash found for client '%s' in round %d",
			clientID, roundIdx,
		)
	}

	return string(submissionBytes), nil
}

// =========================================================================
// QueryAuditTrail — Round-Level Audit Report
// =========================================================================

// QueryAuditTrail generates a comprehensive audit report for a specific
// FL communication round, including all verification records and
// aggregate statistics.
//
// This function is used by regulators and auditors to inspect the
// complete history of model update verifications for any given round.
//
// Parameters:
//   - ctx: Transaction context.
//   - roundIdxStr: FL round index (string).
//   - clientIDsJSON: JSON array of client IDs to query (e.g., '["bank_0","bank_1"]').
//
// Returns:
//   - string: JSON-encoded RoundAuditTrail.
//   - error: nil on success, or an error if the query fails.
func (c *GreenFedContract) QueryAuditTrail(
	ctx contractapi.TransactionContextInterface,
	roundIdxStr string,
	clientIDsJSON string,
) (string, error) {
	roundIdx, err := strconv.Atoi(roundIdxStr)
	if err != nil {
		return "", fmt.Errorf("invalid round index '%s': %v", roundIdxStr, err)
	}

	// Parse the list of client IDs
	var clientIDs []string
	err = json.Unmarshal([]byte(clientIDsJSON), &clientIDs)
	if err != nil {
		return "", fmt.Errorf("invalid client IDs JSON '%s': %v", clientIDsJSON, err)
	}

	// Collect verification records for each client
	var records []VerificationRecord
	totalAccepted := 0
	totalRejected := 0

	for _, clientID := range clientIDs {
		verifyKey := fmt.Sprintf("VERIFY_%s_%d", clientID, roundIdx)
		recordBytes, err := ctx.GetStub().GetState(verifyKey)
		if err != nil {
			continue // Skip clients without verification records
		}
		if recordBytes == nil {
			continue
		}

		var record VerificationRecord
		err = json.Unmarshal(recordBytes, &record)
		if err != nil {
			continue
		}

		records = append(records, record)

		if record.Status == "ACCEPTED" {
			totalAccepted++
		} else {
			totalRejected++
		}
	}

	totalSubmissions := totalAccepted + totalRejected
	acceptanceRate := 0.0
	if totalSubmissions > 0 {
		acceptanceRate = float64(totalAccepted) / float64(totalSubmissions)
	}

	// Build the audit trail report
	auditTrail := RoundAuditTrail{
		RoundIdx:         roundIdx,
		TotalSubmissions: totalSubmissions,
		TotalAccepted:    totalAccepted,
		TotalRejected:    totalRejected,
		AcceptanceRate:   math.Round(acceptanceRate*10000) / 10000,
		Records:          records,
		GeneratedAt:      time.Now().UTC().Format(time.RFC3339),
	}

	auditJSON, err := json.Marshal(auditTrail)
	if err != nil {
		return "", fmt.Errorf("failed to marshal audit trail: %v", err)
	}

	return string(auditJSON), nil
}

// =========================================================================
// Main — Chaincode Entry Point
// =========================================================================

func main() {
	chaincode, err := contractapi.NewChaincode(&GreenFedContract{})
	if err != nil {
		fmt.Printf("Error creating GreenFed-XAI chaincode: %v\n", err)
		return
	}

	if err := chaincode.Start(); err != nil {
		fmt.Printf("Error starting GreenFed-XAI chaincode: %v\n", err)
	}
}

