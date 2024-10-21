package main

type CharacterState struct {
	PlayerNumber  int
	MovementState int32
	AttackState   int32
	Controllable  bool
}

type PlayerInput struct {
	PlayerNumber int
	Inputs       int32
	Facing       float32
}

type FrameData struct {
	PlayerNumber           int
	CurrentMoveFrame       int32
	CurrentMoveReferenceID int64
}

type Meters struct {
	PlayerNumber            int
	LifePercentage          float32
	MeterPercentage         float32
	MeterMax                float32
	DizzyPercentage         float32
	GuardPointsPercentage   float32
	RecoverableHpPercentage float32
}

type Velocity struct {
	PlayerNumber       int
	HorizontalVelocity float32
	VerticalVelocity   float32
}

type AttackHit struct {
	PlayerNumber int
	MoveHit      bool
	MoveGuarded  bool
	ComboCount   int32
}

type Position struct {
	PlayerNumber int
	PositionX    float32
	PositionY    float32
}

type GameState struct {
	Character CharacterState `json:"character"`
	Input     PlayerInput    `json:"input"`
	Frame     FrameData      `json:"frame"`
	Meter     Meters         `json:"meter"`
	Velocity  Velocity       `json:"velocity"`
	Attack    AttackHit      `json:"attack"`
	Position  Position       `json:"position"`
}

func (s *System) GetGameState(charNum int) GameState {
	var iBit InputBits
	if s.chars[charNum] != nil && s.chars[charNum][0] != nil {
		if sys.netInput != nil {
			iBit.SetInput(0)
			//fmt.Printf("%v - %v%v \n", iBit, s.chars[charNum][0].name, i)
			//for o := range sys.inputRemap{
			//	fmt.Printf("Remap%v: %v \n",o, sys.inputRemap[o])
			//}
		} else {
			iBit.SetInput(sys.inputRemap[charNum])
			//fmt.Printf("%v - %v%v \n", iBit, s.chars[charNum][0].name, i)
			//for o := range sys.inputRemap{
			//	fmt.Printf("Remap%v: %v \n",o, sys.inputRemap[o])
			//}
		}
	}
	charState := CharacterState{charNum, int32(s.chars[charNum][0].ss.stateType), int32(s.chars[charNum][0].ss.moveType), s.chars[charNum][0].ctrl()}
	input := PlayerInput{charNum, int32(iBit), s.chars[charNum][0].facing}
	frame := FrameData{charNum, s.chars[charNum][0].ss.time, int64(s.chars[charNum][0].ss.no)}
	meters := Meters{
		charNum,
		float32(s.chars[charNum][0].life) / float32(s.chars[charNum][0].lifeMax),
		float32(s.chars[charNum][0].power) / float32(s.chars[charNum][0].powerMax),
		float32(s.chars[charNum][0].powerMax), float32(s.chars[charNum][0].dizzyPoints) / float32(s.chars[charNum][0].dizzyPointsMax),
		float32(s.chars[charNum][0].guardPoints) / float32(s.chars[charNum][0].guardPointsMax),
		float32(s.chars[charNum][0].redLife) / float32(s.chars[charNum][0].lifeMax),
	}
	velocity := Velocity{charNum, s.chars[charNum][0].vel[0], s.chars[charNum][0].vel[1]}
	attack := AttackHit{charNum, s.chars[charNum][0].moveHit() == 1, s.chars[charNum][0].moveGuarded() == 1, s.lifebar.co[s.chars[charNum][0].teamside].combo}
	pos := Position{charNum, s.chars[charNum][0].pos[0], s.chars[charNum][0].pos[1]}

	// Create a new GameState instance
	gameState := GameState{
		Character: charState,
		Input:     input,
		Frame:     frame,
		Meter:     meters,
		Velocity:  velocity,
		Attack:    attack,
		Position:  pos,
	}

	return gameState
}
