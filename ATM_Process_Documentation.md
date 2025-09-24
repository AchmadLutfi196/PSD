# ATM Cash Withdrawal Process - XPDL Documentation

## Overview
This document describes the XPDL 2.2 compliant ATM Cash Withdrawal Process that addresses the flow connection issues and implements a complete, robust workflow.

## File: ATM_Cash_Withdrawal_Process.xpdl

### Process Structure

#### Participants (4)
1. **CUSTOMER (Nasabah)** - Human participant who uses the ATM
2. **ATM_SYSTEM** - System participant representing ATM hardware/software
3. **BANK_SYSTEM** - System participant for core banking operations
4. **SECURITY_SYSTEM** - System participant for logging and monitoring

#### Lanes (4)
- **Customer Lane** - Customer interactions
- **ATM System Lane** - ATM operations and user interface
- **Bank System Lane** - Banking validations and transactions
- **Security System Lane** - Logging and monitoring activities

### Process Flow

#### Main Happy Path
1. **START_PROCESS** → Customer approaches ATM
2. **INSERT_CARD** → Customer inserts card
3. **READ_CARD** → ATM reads card data
4. **LOG_SESSION_START** → Security system logs session start
5. **VALIDATE_CARD** → Bank validates card
6. **CARD_VALID_GATEWAY** → Decision point for card validity
7. **ENTER_PIN** → Customer enters PIN
8. **VALIDATE_PIN** → Bank validates PIN
9. **PIN_VALID_GATEWAY** → Decision point for PIN validity
10. **DISPLAY_MENU** → ATM shows menu options
11. **SELECT_AMOUNT** → Customer selects withdrawal amount
12. **CHECK_BALANCE** → Bank checks account balance
13. **BALANCE_VALID_GATEWAY** → Decision point for balance sufficiency
14. **PROCESS_TRANSACTION** → Bank processes transaction
15. **DEBIT_ACCOUNT** → Bank debits customer account
16. **DISPENSE_CASH** → ATM dispenses cash
17. **LOG_TRANSACTION** → Security logs successful transaction
18. **PRINT_RECEIPT** → ATM prints receipt
19. **RETURN_CARD_SUCCESS** → ATM returns card
20. **TAKE_CASH_CARD** → Customer takes cash and card
21. **END_SUCCESS** → Process completes successfully

#### Error Handling Flows

##### Card Validation Error
- **CARD_VALID_GATEWAY** → **DISPLAY_CARD_ERROR** → **RETURN_CARD_ERROR** → **END_ERROR**

##### PIN Validation Error
- **PIN_VALID_GATEWAY** → **INCREMENT_PIN_ATTEMPTS** → **PIN_ATTEMPTS_GATEWAY**
  - If < 3 attempts: Return to **ENTER_PIN**
  - If ≥ 3 attempts: **BLOCK_CARD** → **END_CARD_BLOCKED**

##### Insufficient Balance Error
- **BALANCE_VALID_GATEWAY** → **DISPLAY_BALANCE_ERROR** → **END_ERROR**

### Key Features Implemented

#### 1. Fixed Flow Connections ✅
- All 29 activities are properly connected via 30 transitions
- No orphaned or disconnected activities
- Proper start and end points for all paths

#### 2. Gateway Decision Points ✅
- **CARD_VALID_GATEWAY**: Routes based on card validation result
- **PIN_VALID_GATEWAY**: Routes based on PIN validation result  
- **PIN_ATTEMPTS_GATEWAY**: Routes based on number of PIN attempts
- **BALANCE_VALID_GATEWAY**: Routes based on account balance sufficiency

#### 3. Complete Error Handling ✅
- Card rejection handling
- Wrong PIN handling with attempt counting
- Card blocking after 3 wrong attempts
- Insufficient balance handling
- Proper error messaging and card return

#### 4. Security System Integration ✅
- Session start logging
- Transaction completion logging
- Security monitoring throughout process

#### 5. Bank System Integration ✅
- Card validation
- PIN validation
- Balance checking
- Account debiting
- Transaction recording

#### 6. Proper Lane Organization ✅
- Customer actions in customer lane
- ATM operations in ATM lane
- Banking operations in bank lane
- Security operations in security lane

### Data Fields

The process includes proper data fields for:
- **CARD_NUMBER**: Customer card identifier
- **PIN**: Customer PIN
- **AMOUNT**: Withdrawal amount
- **BALANCE**: Account balance
- **TRANSACTION_ID**: Unique transaction identifier
- **CARD_VALID**: Boolean card validation result
- **PIN_VALID**: Boolean PIN validation result
- **SUFFICIENT_BALANCE**: Boolean balance check result
- **PIN_ATTEMPTS**: Counter for PIN attempts

### Graphics and Coordinates

All activities include proper positioning:
- **Customer Lane**: Y-coordinate 50-150
- **ATM Lane**: Y-coordinate 150-250  
- **Bank Lane**: Y-coordinate 250-350
- **Security Lane**: Y-coordinate 350-450
- **Horizontal flow**: X-coordinates from 50 to 2430

### XPDL 2.2 Compliance

The file is fully compliant with XPDL 2.2 specifications:
- ✅ Proper XML namespace declarations
- ✅ Valid XPDL package structure
- ✅ Correct participant definitions
- ✅ Proper activity and transition elements
- ✅ Valid gateway routing elements
- ✅ Appropriate data field definitions
- ✅ Complete graphics information
- ✅ Well-formed XML structure

### Process Validation Results

- **Activities**: 29 (including start/end events)
- **Transitions**: 30 (all activities connected)
- **Gateways**: 4 (all with proper conditions)
- **Participants**: 4 (Customer, ATM, Bank, Security)
- **Lanes**: 4 (proper swimlane organization)
- **Error Activities**: 6 (comprehensive error handling)
- **XML Validation**: ✅ Well-formed

### Solutions to Original Problems

#### Fixed Activity Tag Matching Issues
- All activities have properly matched start and end tags
- No line 293/302 type errors (activities properly closed)
- Consistent XML structure throughout

#### Fixed Broken Transitions
- All transitions have valid From/To references
- No orphaned activities
- Proper flow continuity

#### Added Missing Error Handling
- Card rejection flow
- PIN error flow with retry logic
- Balance error flow
- Card blocking mechanism

#### Ensured Consistent Flow Logic
- Logical progression from start to end
- Proper decision points
- Clear error paths
- Appropriate process termination

#### Fixed Coordinates and Graphics
- All activities have positioning information
- Proper lane organization
- Consistent visual layout
- Accurate connector positioning

This XPDL file provides a complete, robust, and compliant ATM cash withdrawal process that addresses all the issues mentioned in the original problem statement.