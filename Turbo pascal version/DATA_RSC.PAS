{DATA INPUT RESOURCE}

 UNIT Data_rsc;

  {* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * }


  INTERFACE

    USES  Setts;
    {Provides all file-based operations}
    {Contains Global variables, types & constants, relevent to the Data Input Resource}

    {Variable names mapped to unique variable numbers, as passed by WAVE}
      TYPE
        Namestr = String[6];
      CONST
        ngrvar=10;   ngcvar=21;    nstvar=21;   nscvar=11;
        ngmxvar=1;   nsmxvar=8;   ngacvar=3;   nsacvar=1;  nmixvar=4;
        grnam0=0;    gcnam0=20;   stnam0=50;   scnam0=100;
        gmxnam0=150; smxnam0=175; gacnam0=200; sacnam0=225;
        mixnam0=235;
        FormName=255;

        VFuncName : ARRAY [0..5]  OF String[3]
            = ('','MX','mn','Avg','Abs','AMX');

        VarNames :
          RECORD
            grvar  : ARRAY [1..ngrvar]  OF String[6];
            gcvar  : ARRAY [1..ngcvar]  OF String[6];
            stvar  : ARRAY [1..nstvar]  OF String[6];
            scvar  : ARRAY [1..nscvar]  OF String[6];
            gmxvar : ARRAY [1..ngmxvar] OF String[6];
            smxvar : ARRAY [1..nsmxvar] OF String[6];
            gacvar : ARRAY [1..ngacvar] OF String[6];
            sacvar : ARRAY [1..nsacvar] OF String[6];
            mixvar : ARRAY [1..nmixvar] OF String[6];
          END
            =  ( grvar  : ( 'XVEL','YVEL','S11','S22','S12','ZVEL','S33','S23','S31','S31' );
                 gcvar  : ( 'DIL','V-abs','TauMx','Sig-3','Sig-1','Sig-2','M-Ang1','ESS','DEV',
                            'aXVEL','aYVEL','aS12','aZVEL','aS23','aS31','aS31',
                            'S.E.','K.E.','T.E.','Fail','e-plas');
                 stvar  : ( 't11_s1','t11_s0','t1v_s1','t1v_s0','t1sh','t1d-r',
                            'nvel-r','ndis-r','Sh-bnd','T1slip','TaSlip','N-bnd',
                            't22_s1','t22_s0','t2v_s1','t2v_s0','t1t2s1',
                            't1t2s0','t2sh','t2d-r','T2slip');
                 scvar  :  ( 'nd-s1','nd-s0','Tad-r','t1d_s1','t1d_s0','t2d_s1','t2d_s0',
                             'stv_s1','stv_s0','sts_s1','sts_s0');
                 gmxvar :  ( 'MaxVel' );
                 smxvar :  ( 'nv-mx','nd-mx','t1v-mx','t1d-mx','t2v-mx','t2d-mx',
                             't11-mx','t22-mx' );
                 gacvar :  ( 'Xdisp','Ydisp','Zdisp' );
                 sacvar :  ( '' );
                 mixvar :  ( 'skn', 'sks', 'sNs', 'sSs' )   );
        VarPositions :
          RECORD
            grvar  : ARRAY [1..ngrvar]  OF Byte;
            gcvar  : ARRAY [1..ngcvar]  OF Byte;
            stvar  : ARRAY [1..nstvar]  OF Byte;
            scvar  : ARRAY [1..nscvar]  OF Byte;
            gmxvar : ARRAY [1..ngmxvar] OF Byte;
            smxvar : ARRAY [1..nsmxvar] OF Byte;
            gacvar : ARRAY [1..ngacvar] OF Byte;
            sacvar : ARRAY [1..nsacvar] OF Byte;
            mixvar : ARRAY [1..nmixvar] OF Byte;
          END
            =  ( grvar  : (1,2,0,0,3,4,0,6,5,5);
                 gcvar  : (0,0,0,0,3,3,0,0,0,0,0,0,0,0,0,0,0,0,0, 3,3);
                 stvar  : (0,0,8,8,8,8,  0,0,8,8,0,0,  0,0,9,9,10,  10,9,9,9 );
                 scvar  : (0,0,0,0,0,0,0,0,0,0,0);
                 gmxvar : (0);
                 smxvar : (0,0,0,0,0,0,0,0);
                 gacvar : (0,0,0);
                 sacvar : (0);
                 mixvar : (0,0,0,0)  );

    CONST
      MaxSnap      = 250; {350}
      MaxHist      = 510;
{      MaxSnap      = 400;}
{      MaxHist      = 350;}
      MaxPlots     =  6;
      MaxDump      =  50;
      MaxGeom      =   5;
      MaxTraj      = MaxHist;
{      MaxHistSz    = 16380;} {largest}
      MaxHistSz    = 4096;
{      MaxHistSz    = 8192;}
{      MaxHistSz    = 4096;} {std}
{      MaxHistSz    = 2048;}
      MaxCrackD    =   100;

          {To allow more waveforms etc}
(*          MaxSnap      =  30;
          MaxHist      =  200;
          MaxPlots     =  10;
          MaxDump      =  20;
          MaxGeom      =  10;
          MaxTraj      = MaxHist;
          MaxHistSz    = 1624;
          MaxCrackD    =   100; *)


      Maxprops     =  10;
      Maxmats      =  10;
      MaxSources   =  100;
      MaxStopes    =  1560;
      MaxDumpSz    =  51;
      SXtra1       = 5; {Extra Real values at start of file, besides snapshot}
      SXtra2       = 0; {Extra Real values at end of file}
      DXtra1       = 2; {Extra Real values at start of file, besides the dump/snapshot}
      DXtra2       = 1; {Extra Real values at end of file}
      HXtra1       = 4; {Extra Real values at start of file, besides history}
      HXtra2       = 4; {Extra Real values at end of file}


    TYPE
      Axes        = (X,Y);
      Point       = ARRAY [Axes] OF Single;
      Vector      = ARRAY [Axes] OF Single;
      PropRec     = RECORD
                      {Values passed from WAVE}
                         Bulk,Shear,Dens : Single;
                    END;
      MatRec      = RECORD
                      {Values passed from WAVE}
                         MatNum        : LongInt;
                         i1,i2         : LongInt;
                         j1,j2         : LongInt;
                         k1,k2         : LongInt;
                    END;
      SourceRec   = RECORD
                      {Values passed from WAVE}
                         stype         : LongInt;
                         i1,i2         : LongInt;
                         j1,j2         : LongInt;
                         k1,k2         : LongInt;
                    END;
      StopeOutline = RECORD
                      Qty, TotPt, Pos1, N1 : LongInt;
                      Pt : ARRAY [1..2,1..1000] OF LongInt;
                     END;
      StopeLayout = ARRAY [1..$FFFF] OF Byte;
      StopeRec    = RECORD
                      {Values passed from WAVE}
                         stopID, stype : LongInt;
                         i1,i2         : LongInt;
                         j1,j2         : LongInt;
                         k1,k2         : LongInt;
                      Outline        : ^StopeOutline;
                      Layout         : ^StopeLayout;
                      Laysz          : Word;
                    END;
      SnapMapRec  = RECORD
                      Offset        : LongInt;
                      {Values passed from WAVE}
                         SnapVar       : LongInt;
                         t1,Time       : Single;
                         i1,i2         : LongInt;
                         j1,j2         : LongInt;
                         k1,k2         : LongInt;
                         Ax1, Ax2      : Char;      {characters identifying axes 1 and 2}
                         dx, dy        : Single;    {step length for axes 1 and 2}
                         sx, sy        : LongInt;   {sample freq for axes 1 and 2}
                         Xqty, Yqty    : LongInt;
                         MaxVal,MinVal : Single;
                         gnum, gID     : LongInt;   {ID of geometry for this snapshot}
                      QuickX, QuickY   : Word;
                      SFunc            : Word;
{                      TrueGscale       : Boolean;}
                    END; {SnapMapRec}
      DumpMapRec  = RECORD
                      Offset        : LongInt;
                      {Values passed from WAVE}
                         Time          : Single;
                         i1,i2         : LongInt;
                         j1,j2         : LongInt;
                         k1,k2         : LongInt;
                         Ax1, Ax2      : Char;      {characters identifying axes 1 and 2}
                         dx, dy        : Single;    {step length for axes 1 and 2}
                         sx, sy        : LongInt;   {sample freq for axes 1 and 2}
                         iqty, jqty    : LongInt;
                         VQty          : LongInt;
                         gnum, gID     : LongInt;   {ID of geometry for this snapshot}
                         dz            : Single;
                         sz, kqty      : LongInt;
                      Dim3          : Boolean;
                      MaxVal        : Single;
                      Xqty, Yqty    : LongInt;
                      QuickX, QuickY: Word;
                    END; {DumpMapRec}
      HistMapRec  = RECORD
                      Offset        : LongInt;
                      {Values passed from WAVE}
                         HistVar       : LongInt;
                         t1,t2         : Single;
                         i1,i2         : LongInt;
                         j1,j2         : LongInt;
                         k1,k2         : LongInt;
                         Ax1           : Char;      {character identifying axis 1}
                         dx            : Single;    {step length for axis 1}
                         sx            : LongInt;   {sample freq for axis 1}
                         Xqty          : LongInt;
                         MaxVal,MinVal : Single;
                         gnum, gID     : LongInt;   {ID of geometry for this snapshot}
                      Tqty             : LongInt;
                      Xp, Yp, Zp       : LongInt;
                      Tstart, Time, dT : Single;
                      VMax             : Single;
                      HFunc            : Word;
                    END; {HistMapRec}
      DumpRec    = RECORD
                     CASE DumpType OF
                       Vvect    :   (V     : Vector);
                       Stress   :   (Major, Minor, Ang : Single);
                   END;
      Outline3D  = RECORD
                     Qty, TotPt, Pos1, N1 : LongInt;
                   END;
      GeomMapRec = RECORD
                      Offset        : LongInt;
                      {Values passed from WAVE}
                         gnum, gID : LongInt;
                         proptotT, mattotT, sourcetotT, stopetotT : LongInt;
                         time          : single;
                         ntot          : LongInt;
                         i1,i2         : LongInt;
                         j1,j2         : LongInt;
                         k1,k2         : LongInt;
                         dx,dy,dz      : single;
                         GRlen         : LongInt;
                      Model3D        : Boolean;
                      COG_i,COG_j,COG_k : integer;
                      proptot, mattot, sourcetot, stopetot, PropPos, MatPos : LongInt;
                    END;
      TrajPoint = RECORD
                    i1,j1,k1   : Integer;   {position in grid}
                    his        : ARRAY [0..2] OF Integer;   {associated history for xvel, yvel and zvel}
                  END;
      CrackDMapRec = RECORD
                      Offset        : LongInt;
                      {Values passed from WAVE}
                         Time            : Single;
                         ic1,ic2         : LongInt;
                         gnum, gID       : LongInt;  {ID of geometry}
                         StQty,Tqty,VQty : LongInt;  {Qty of cracks, crack elems, and Vars}
                         CrLen           : LongInt;  {Length, entry in crack file}
                    END; {CrackMapRec}
      CrackRec    = RECORD
                      {Values passed from WAVE}
                      dum1                  : single;
                      vNs1,vNs0,vT1s1,vT1s0 : single;
                      uNs1,uNs0,uT1s1,uT1s0 : single;
                      sNs1,sT1s1,sT1s0,sT1sh : single;
                      T1slp,Taslp,Nbnd,Shbnd: single;
                      dum2                  : single;
                    END;

      SnapArray   = ARRAY [1..MaxX, 1..MaxY] OF Single;
      DumpArray   = ARRAY [1..MaxDumpSz, 1..MaxDumpSz] OF DumpRec;
      HistArray   = ARRAY [1..MaxHistSz] OF Single;
      TrajArray   = ARRAY [1..16000] OF Single;
      SnapPtr     = ^SnapArray;
      DumpPtr     = ^DumpArray;
      HistPtr     = ^HistArray;
      TrajPtr     = ^TrajArray;
      SnapMapArray = ARRAY [1..MaxSnap] OF SnapMapRec;
      DumpMapArray = ARRAY [1..MaxDump] OF DumpMapRec;
      HistMapArray = ARRAY [1..MaxHist] OF HistMapRec;
      GeomMapArray = ARRAY [1..MaxGeom] OF GeomMapRec;
      CrackDMapArray = ARRAY [1..MaxCrackD] OF CrackDMapRec;

      ScaleType   = (LinScal, AbsLScal, LogScal, LogLogScal);
      GRrec       = RECORD
                      LegMax     : Single;
                      ScalT      : ScaleType;
                      Qty, St    : Integer;
                      Ds_Voff,
                        Pl_Voff  : Integer;
                      xstr       : String[2];
                      h0         : Integer;
                      Hoff       : single;
                    END; {GRrec}
      SeismRec    = RECORD
                      Hnum    : Word;
                      Hstmap  : HistMapRec;
                      Hist    : HistPtr;
                      Form    : String;
                      FormSet : Boolean;
                      GR      : GRrec;
                    END; {SeismRec}
      StopeType    = ARRAY [1..MaxStopes]  OF StopeRec;

    VAR
      MapFile      : File;
      SnapFile     : File OF Single;
      DumpFile     : File OF Single;
      HistFile     : File OF Single;
      GeomFile     : File;
      CrackFile    : File;
      OutlineFile  : Text;
      SnapMap      : ^SnapMapArray;
      DumpMap      : ^DumpMapArray;
      HistMap      : ^HistMapArray;
      GeomMap      : ^GeomMapArray;
      CrackDMap    : ^CrackDMapArray;
{      GeomMap      : ARRAY [1..MaxGeom] OF GeomMapRec;}
      TrajMap      : RECORD
                        map  : RECORD
                                 Tqty       : Word;
                                 t1, t2, dt : single;
                               END;
                        list : ARRAY [1..MaxTraj] OF TrajPoint;
                      END;

      NoSnap,NoDump,NoHist,NoGeom,NoCrackD              : Boolean;
      QtySnap,QtyHist,QtyDump,QtyGeom,QtyTraj,QtyCrackD : Word;
      NoOutline    : Boolean;
      SnapShot     : SnapPtr;
      Dump         : DumpPtr;
      Props        : ARRAY [1..MaxProps]   OF PropRec;
      Mats         : ARRAY [1..MaxMats]    OF MatRec;
      Sources      : ARRAY [1..MaxSources] OF SourceRec;
      Stopes       : ^StopeType;
      Seism        : ARRAY [1..MaxSeism]  OF SeismRec;
      Level        : ARRAY [1..MaxLevels] OF Single;
      LevCol       : ARRAY [1..MaxLevels] OF Byte;
      ImageFname   : String;
      CurSnapRec   : RECORD
                       SnapVar        : Integer;
                       Xqty, Yqty     : LongInt;
                       MaxVal,MinVal  : Single;
                       QuickX, QuickY : Word;
                       Time           : Single;
                       Ax1, Ax2       : Char;
                       Snp            : Word;
                       Vpos           : Byte;
                       SFunc          : Word;
                     END; {CurSnapRec}
      Traj         : RECORD
                       Nt, Np   : Word; {Number of trajs and points in each}
                       MemSz    : Word;
                       Tstp     : single;
                       MaxV     : single;
                       Data     : TrajPtr;
                       NotRead  : Boolean;
                       Trtype   : Integer;  {0-displ; 1-vel; 2-accel}
                       DScale   : Boolean;  {Scale on distance from source}
                     END;

    FUNCTION VarName (Vnum : Byte) : NameStr;

    FUNCTION VarPos (Vnum : Byte) : Byte;

    PROCEDURE DI_ListFiles (Path, Mask : String);

    PROCEDURE DI_RqstPath;

    FUNCTION DI_FileInit : Boolean;

    PROCEDURE DI_AssignFile (Name : FileName);

    FUNCTION DI_AssignF (Fname : String) : Boolean;

    PROCEDURE DI_RqstImageFile (VAR Num : Word);

    PROCEDURE DI_Close;

    PROCEDURE DI_ReadMap;

    PROCEDURE DI_ReadSnap (S: Word);

    PROCEDURE DI_ExtractSnap;

    PROCEDURE DI_ReadDump (D: Word);
    PROCEDURE DI_ReadDump2 (D: Word);

    PROCEDURE DI_DumpContour (D: Word);

    PROCEDURE DI_ReadHist (S: Word; H: HistPtr);

    PROCEDURE DI_ReadGeom (VAR G: Word;  VAR Gnum, gID  : LongInt);

    PROCEDURE DI_ClearGeom (G: Word);

    PROCEDURE DI_CalcTraject;

    PROCEDURE DI_ReadTraject (G:integer);

    PROCEDURE DI_ClearTraject;

    PROCEDURE DI_CvertPSS (Qty : Integer);

    PROCEDURE DI_CvertAscii (Qty, ityp : Integer);

    PROCEDURE DI_InitVar;

    PROCEDURE DI_ClearVar;


  IMPLEMENTATION

    USES  Dos, Crt, Funct, Graph, ScrnBMP;

    FUNCTION VarName (Vnum : Byte) : NameStr;
      BEGIN
        CASE Vnum OF
          (grnam0+1)..gcnam0   : VarName := VarNames.grvar  [Vnum-grnam0];
          (gcnam0+1)..stnam0   : VarName := VarNames.gcvar  [Vnum-gcnam0];
          (stnam0+1)..scnam0   : VarName := VarNames.stvar  [Vnum-stnam0];
          (scnam0+1)..gmxnam0  : VarName := VarNames.scvar  [Vnum-scnam0];
          (gmxnam0+1)..smxnam0 : VarName := VarNames.gmxvar [Vnum-gmxnam0];
          (smxnam0+1)..gacnam0 : VarName := VarNames.smxvar [Vnum-smxnam0];
          (gacnam0+1)..sacnam0 : VarName := VarNames.gacvar [Vnum-gacnam0];
          (sacnam0+1)..mixnam0 : VarName := VarNames.sacvar [Vnum-sacnam0];
          (mixnam0+1)..254     : VarName := VarNames.mixvar [Vnum-mixnam0];
          FormName             : VarName := 'Form';
        END {Case}
      END; {VarName}

    FUNCTION VarPos (Vnum : Byte) : Byte;
      BEGIN
        CASE Vnum OF
          (grnam0+1)..gcnam0   : VarPos := VarPositions.grvar  [Vnum-grnam0];
          (gcnam0+1)..stnam0   : VarPos := VarPositions.gcvar  [Vnum-gcnam0];
          (stnam0+1)..scnam0   : VarPos := VarPositions.stvar  [Vnum-stnam0];
          (scnam0+1)..gmxnam0  : VarPos := VarPositions.scvar  [Vnum-scnam0];
          (gmxnam0+1)..smxnam0 : VarPos := VarPositions.gmxvar [Vnum-gmxnam0];
          (smxnam0+1)..gacnam0 : VarPos := VarPositions.smxvar [Vnum-smxnam0];
          (gacnam0+1)..sacnam0 : VarPos := VarPositions.gacvar [Vnum-gacnam0];
          (sacnam0+1)..mixnam0 : VarPos := VarPositions.sacvar [Vnum-sacnam0];
          (mixnam0+1)..254     : VarPos := VarPositions.mixvar [Vnum-mixnam0];
          FormName             : VarPos := 0;
        END {Case}
      END; {VarPos}


    PROCEDURE DI_ListFiles (Path, Mask : String);
      VAR
        DirInfo : SearchRec;
        n       : word;
      BEGIN
        Writeln (wr_file);
        Writeln (wr_file,'Directory ' + Path + ', contains the following plot files:' : 10);
        Writeln (wr_file);
        n := 0;
        FindFirst (Path+'\*.*', $10, DirInfo);
        WHILE (DosError = 0) DO
          BEGIN
            IF ( (DirInfo.Attr AND $10) > 0) THEN
               BEGIN
                 Write (wr_file, '    < '+DirInfo.Name+' >':18);
                 Inc(n);
                 IF ((n MOD 4)=0)
                    THEN BEGIN writeln(wr_file); wr_flush(wr_nohold2); END;
               END;
            FindNext(DirInfo);
          END;
        FindFirst (Path+'\'+Mask, AnyFile, DirInfo);
        WHILE (DosError = 0) DO
          BEGIN
            Write (wr_file, DirInfo.Name:18); Inc(n);
            IF ((n MOD 4)=0) THEN BEGIN writeln(wr_file); wr_flush(wr_nohold2); END;
            FindNext(DirInfo);
          END;
        Writeln(wr_file); wr_flush(wr_hold);
      END; {DI_ListFiles}

     PROCEDURE DI_RqstPath;
       BEGIN
         Writeln (wr_file,'Set the drive & directory containing plot files & image files.');
         Writeln (wr_file);
         Writeln (wr_file,'    Enter Drive:     ');
         wr_flush(wr_nohold);
         Sets.Drive := ReadKey;
{         Writeln (wr_file,Sets.Drive);}
         OutText (Sets.Drive);
         Writeln (wr_file,'    Enter Directory: ');
         wr_flush(wr_nohold);
         ReadLn (Sets.Dir);
       END; {DI_RqstPath}

    FUNCTION DI_FileInit : Boolean;
      VAR
        NO_error : Boolean;
        InpName  : String;
        Ch       : Char;
      BEGIN
{        ClrScr;}
        NO_error := True;
        {$I-} {Turn off IO error checking}
        REPEAT
          WHILE (IOResult <> 0) DO; {Clear IOResult}
          IF (NOT NO_error)
             THEN BEGIN
                    wr_clear;
                    Writeln (wr_file,'Errors in the specified file or path - please re-enter');
                    DI_RqstPath;
                  END;
          NO_error := True;
          DI_ListFiles (Sets.Drive + ':' + Sets.Dir, '*.MAP');
          Write (wr_file,'    Enter File Name (', Sets.FName, '): ');
          wr_flush(wr_hold);
          Ch := ReadKey;
          OutText(Ch);
          IF ( Ch = #27)
             THEN DI_FileInit := False
             ELSE BEGIN
                    IF NOT ( Ch IN [#13,#0] )
                       THEN BEGIN
                              ReadLn (InpName);
                              Sets.FName := Ch + InpName;
                            END;
                    ASSIGN (MapFile ,     (Sets.Drive + ':' + Sets.Dir + '\' + Sets.FName + '.MAP') );
                    NO_error := NO_error AND (IOResult = 0);
                    RESET  (MapFile,1);
                    NO_error := NO_error AND (IOResult = 0);

                    ASSIGN (SnapFile,     (Sets.Drive + ':' + Sets.Dir + '\' + Sets.FName + '.SNP') );
                    NO_error := NO_error AND (IOResult = 0);
                    RESET  (SnapFile);
                    NoSnap   := (IOResult <> 0);

                    ASSIGN (DumpFile,     (Sets.Drive + ':' + Sets.Dir + '\' + Sets.FName + '.DMP') );
                    NO_error := NO_error AND (IOResult = 0);
                    RESET  (DumpFile);
                    NoDump   := (IOResult <> 0);

                    ASSIGN (HistFile ,    (Sets.Drive + ':' + Sets.Dir + '\' + Sets.FName + '.HST') );
                    NO_error := NO_error AND (IOResult = 0);
                    RESET  (HistFile);
                    NoHist   := (IOResult <> 0);

                    ASSIGN (GeomFile ,    (Sets.Drive + ':' + Sets.Dir + '\' + Sets.FName + '.GEO') );
                    NO_error := NO_error AND (IOResult = 0);
                    RESET  (GeomFile,1);
                    NoGeom   := (IOResult <> 0);

                    ASSIGN (OutLineFile ,    (Sets.Drive + ':' + Sets.Dir + '\' + Sets.FName + '.OTL') );
                    NO_error := NO_error AND (IOResult = 0);
                    RESET  (OutLineFile);
                    NoOutline := (IOResult <> 0);

                    ASSIGN (CrackFile ,    (Sets.Drive + ':' + Sets.Dir + '\' + Sets.FName + '.CRK') );
                    NO_error := NO_error AND (IOResult = 0);
                    RESET  (CrackFile);
                    NoCrackD := (IOResult <> 0);

                    IF (No_error)
                       THEN DI_FileInit := True;
                  END;
        UNTIL (NO_error);
        {$I+} {Turn on IO error checking}
        InitGraf2BMP (Sets.Drive + ':' + Sets.Dir + '\');
      END; {DI_FileInit}

{    FUNCTION DI_AssignFile (Name : FileName) : Boolean;}
    PROCEDURE DI_AssignFile (Name : FileName);
      VAR
        NO_error : Boolean;
      BEGIN
        NO_error := True;
        {$I-} {Turn off IO error checking}
          ASSIGN (MapFile ,     (Sets.Drive + ':' + Sets.Dir + '\' + Name + '.MAP') );
          NO_error := NO_error AND (IOResult = 0);
          RESET  (MapFile,1);
          NO_error := NO_error AND (IOResult = 0);

          ASSIGN (SnapFile,     (Sets.Drive + ':' + Sets.Dir + '\' + Name + '.SNP') );
          NO_error := NO_error AND (IOResult = 0);
          RESET  (SnapFile);
          NoSnap   := (IOResult <> 0);

          ASSIGN (DumpFile,     (Sets.Drive + ':' + Sets.Dir + '\' + Name + '.DMP') );
          NO_error := NO_error AND (IOResult = 0);
          RESET  (DumpFile);
          NoDump   := (IOResult <> 0);

          ASSIGN (HistFile ,    (Sets.Drive + ':' + Sets.Dir + '\' + Name + '.HST') );
          NO_error := NO_error AND (IOResult = 0);
          RESET  (HistFile);
          NoHist   := (IOResult <> 0);

          ASSIGN (GeomFile ,    (Sets.Drive + ':' + Sets.Dir + '\' + Name + '.GEO') );
          NO_error := NO_error AND (IOResult = 0);
          RESET  (GeomFile,1);
          NoGeom   := (IOResult <> 0);

          ASSIGN (OutLineFile ,    (Sets.Drive + ':' + Sets.Dir + '\' + Name + '.OTL') );
          NO_error := NO_error AND (IOResult = 0);
          RESET  (OutLineFile);
          NoOutline := (IOResult <> 0);

          ASSIGN (CrackFile ,    (Sets.Drive + ':' + Sets.Dir + '\' + Name + '.CRK') );
          NO_error := NO_error AND (IOResult = 0);
          RESET  (CrackFile);
          NoCrackD := (IOResult <> 0);

          {DI_AssignFile:= No_error;}
        {$I+} {Turn on IO error checking}
      END; {DI_AssignFile}

    FUNCTION DI_AssignF (Fname : String) : Boolean;
      VAR
        NO_error : Boolean;
      BEGIN
        NO_error := True;
        {$I-} {Turn off IO error checking}
          ASSIGN (MapFile ,     (FName + '.MAP') );
          NO_error := NO_error AND (IOResult = 0);
          RESET  (MapFile,1);
          NO_error := NO_error AND (IOResult = 0);

          ASSIGN (SnapFile,     (FName + '.SNP') );
          NO_error := NO_error AND (IOResult = 0);
          RESET  (SnapFile);
          NoSnap   := (IOResult <> 0);

          ASSIGN (DumpFile,     (FName + '.DMP') );
          NO_error := NO_error AND (IOResult = 0);
          RESET  (DumpFile);
          NoDump   := (IOResult <> 0);

          ASSIGN (HistFile ,    (FName + '.HST') );
          NO_error := NO_error AND (IOResult = 0);
          RESET  (HistFile);
          NoHist   := (IOResult <> 0);

          ASSIGN (GeomFile ,    (FName + '.GEO') );
          NO_error := NO_error AND (IOResult = 0);
          RESET  (GeomFile,1);
          NoGeom   := (IOResult <> 0);

          ASSIGN (OutLineFile ,    (FName + '.OTL') );
          NO_error := NO_error AND (IOResult = 0);
          RESET  (OutLineFile);
          NoOutline := (IOResult <> 0);

          ASSIGN (CrackFile ,    (FName + '.CRK') );
          NO_error := NO_error AND (IOResult = 0);
          RESET  (CrackFile);
          NoCrackD := (IOResult <> 0);

          DI_AssignF:= No_error;
        {$I+} {Turn on IO error checking}
      END; {DI_AssignF}

    PROCEDURE DI_RqstImageFile (VAR Num : Word);
      VAR
        NoError : Boolean;
      BEGIN
        Writeln (wr_file, '* * * List of Image Files * * *');
        DI_ListFiles (Sets.Drive + ':' + Sets.Dir, '*.I00');
        Write (wr_file, '    Enter Image File Name (No extension): ');
        wr_flush(wr_hold); ReadLn (ImageFName);
        IF (Num <> 0)
           THEN REPEAT
                  Write (wr_file, '* * * List of Image Numbers * * *');
                  DI_ListFiles (Sets.Drive + ':' + Sets.Dir, ImageFName + '.I*');
                  Write (wr_file, 'Enter number of saved image [0..100]: ');
                  wr_flush(wr_hold); Readln (Num);
                UNTIL (Num IN [0..99]);
      END; {DI_RqstImageFile}

    PROCEDURE DI_Close;
      BEGIN
        {$I-} {Turn off IO error checking}
          CLOSE (MapFile);
          CLOSE (SnapFile);
          CLOSE (DumpFile);
          CLOSE (HistFile);
          CLOSE (GeomFile);
          CLOSE (OutlineFile);
          CLOSE (CrackFile);
          WHILE (IOResult <> 0) DO; {Clear IOResult}
        {$I+} {Turn on IO error checking}
      END; {DI_Close}

    PROCEDURE DI_ReadMap;
      VAR
        SnapStart,DumpStart,HistStart,GeomStart,CrackDStart : LongInt;
        DumInt    : LongInt;
        PlType    : LongInt;
        i         : Integer;
      BEGIN
        QtySnap:=0; QtyDump:=0; QtyHist:=0; QtyGeom:=0; QtyCrackD:=0;
        SnapStart:=0; DumpStart:=0; HistStart:=0; GeomStart:=0; CrackDStart:=0;
        WHILE (NOT (EOF(MapFile)) ) DO
          BEGIN
            BlockRead (MapFile, DumInt, 4);
            BlockRead (MapFile, PlType, 4);
            CASE PlType OF
              0 : BEGIN
                    IF (QtySnap<MaxSnap) THEN Inc (QtySnap);
                    WITH SnapMap^ [QtySnap] DO
                      BEGIN
                        BlockRead (MapFile, SnapVar, 78);
                        Seek (MapFile, FilePos(MapFile)+42);
                        SFunc := SnapVar DIV $FFFF; SnapVar := SnapVar MOD $FFFF;
                        Offset    := SnapStart;
                        SnapStart := SnapStart + Xqty * (Yqty+2);
                        CASE Ax1 OF
                          'i', 'I' : BEGIN
{                                       Xst}
                                     END;
                        END;
                      END;
                  END;
              1 : BEGIN
                    IF (QtyHist<MaxHist) THEN Inc(QtyHist);
                    WITH HistMap^ [QtyHist] DO
                      BEGIN
                        BlockRead (MapFile, HistVar, 65);
                        Seek (MapFile, FilePos(MapFile)+55);

                        HFunc := HistVar DIV $FFFF; HistVar := HistVar MOD $FFFF;
                        Tqty := Xqty; Xp := i1; Yp := j1; Zp := k1;
                        CASE Ax1 OF
                          'i', 'I' : BEGIN  Tstart := (i1-1)*dx; Time := (i2-1)*dx;   END;
                          'j', 'J' : BEGIN  Tstart := (j1-1)*dx; Time := (j2-1)*dx;   END;
                          'k', 'K' : BEGIN  Tstart := (k1-1)*dx; Time := (k2-1)*dx;   END;
                          ELSE  BEGIN  Tstart := t1; Time := t2; dT := dx;  END;
                        END; {Case}
                        {dT := dx;}
                        Dt   := (Time-Tstart) / (Tqty-1);

                        {If Tqty > MaxHistSz, then truncate (simpler than mapping)}
                        IF (Xqty > MaxHistSz)
                           THEN BEGIN Tqty := MaxHistSz; Time := Tstart+Tqty*Dt; END;

                        VMax := Max ( ABS(MaxVal), ABS(MinVal) );
                        Offset    := HistStart;
                        HistStart := HistStart + Xqty*3;
                      END;
                  END;
              2 : BEGIN
                    IF (QtyDump<MaxDump) THEN Inc (QtyDump);
                    WITH DumpMap^ [QtyDump] DO
                      BEGIN
                        BlockRead (MapFile, Time, 78);
                        Seek (MapFile, FilePos(MapFile)+42);
                        IF (Vqty = 9)
                           THEN Dim3 := True
                           ELSE Dim3 := False;
                        Offset    := DumpStart;
                        IF (kqty=0) THEN BEGIN  kqty:=1; sz:=1; dz:=dx;  END;  {needed for b-compatibility}
                        DumpStart := DumpStart + 1 + iqty*jqty*kqty * Vqty + 1;
                      END;
                  END;
              3 : BEGIN
                    IF (QtyGeom<MaxGeom) THEN Inc (QtyGeom);
                    WITH GeomMap^ [QtyGeom] DO
                      BEGIN
                        BlockRead (MapFile, gnum, 76);
                        Seek (MapFile, FilePos(MapFile)+44);
                        Offset := GeomStart;
                        IF (GRlen = 0)
                           THEN GRlen := (proptotT + mattotT + sourcetotT + stopetotT) * 128; {b-compat}
                        IF (StopeTotT <= MaxStopes) THEN StopeTot := StopeTotT ELSE StopeTot := MaxStopes;
                        IF (propTotT  <= MaxProps)  THEN PropTot  := PropTotT  ELSE PropTot  := MaxProps;
                        IF (matTotT   <= Maxmats)   THEN MatTot   := MatTotT   ELSE MatTot := MaxMats;
                        IF (sourceTotT <= MaxSources) THEN SourceTot := SourceTotT ELSE SourceTot := MaxSources;
                        IF (dx = 0) THEN BEGIN  dx:=1;dy:=1;dz:=1;  END; {b-compat}
                           {Can do this directly as 0:LongInt = 0.0:single; otherwise base a LongInt pointer at dx}
                        GeomMap^[QtyGeom].Model3D := NOT (GeomMap^[QtyGeom].k2 = GeomMap^[QtyGeom].k1);
                        GeomStart := GeomStart + GRlen;
                      END; {With}
                  END;
              4 : BEGIN
                    IF (QtyCrackD<MaxCrackD) THEN Inc(QtyCrackD);
                    WITH CrackDMap^[QtyCrackD] DO
                      BEGIN
                        BlockRead (MapFile, Time, 36);
                        Seek (MapFile, FilePos(MapFile)+84);
                        Offset      := CrackDStart;
                        CrackDStart := CrackDStart+CrLen;
                      END;
                  END;
            END; {Case}
          END; {While}

        IF (NoSnap) THEN QtySnap := 0;
        IF (NoDump) THEN QtyDump := 0;
        IF (NoHist) THEN QtyHist := 0;
        IF (NoGeom) THEN QtyGeom := 0;
        IF (NoCrackD) THEN QtyCrackD := 0;
      END; {DI_ReadMap}


    FUNCTION GetVarPos ( Vnum : Byte;
                         Ax1, Ax2 : Char ) : Byte;
        {Altered for plane of snapshot}
      VAR
        Vpos, Npos : Byte;
      BEGIN
        Vpos := VarPos(Vnum);
        If (Vpos = 8)
           THEN IF (Ax1 IN ['i','I'])
                   THEN Vpos := 1
                   ELSE Vpos := 2;
        If (Vpos = 9)
           THEN IF ([Ax1,Ax2] * ['k','K'] = [])
                   THEN Vpos := 2
                   ELSE Vpos := 4;
        If (Vpos = 10)
           THEN Npos := 3
           ELSE BEGIN
                  Npos := 0;
                  CASE Ax1 OF
                    'i', 'I' : Npos := (Vpos MOD 2);
                    'j', 'J' : Npos := (Vpos MOD 4) DIV 2;
                    'k', 'K' : Npos := (Vpos DIV 4);
                  END;
                  CASE Ax2 OF
                    'i', 'I' : Npos := Npos + (Vpos MOD 2) * 2;
                    'j', 'J' : Npos := Npos + (Vpos MOD 4 DIV 2) * 2;
                    'k', 'K' : Npos := Npos + (Vpos DIV 4) * 2;
                  END;
                END;
        GetVarPos := Npos;
      END; {GetVarPos}


    PROCEDURE DI_ReadSnap (S: Word);
      VAR
        ix, iy, Xpos, Ypos  : Word;
        NextX               : Word;
        Xratio, Yratio      : Single;
        Count               : ARRAY [1..MaxX] OF Integer;
        MapOn               : Boolean;
        SnapF               : File;
        SnapVal             : ARRAY [0..1601] OF Single;
      BEGIN
        ASSIGN (SnapF,     (Sets.Drive + ':' + Sets.Dir + '\' + Sets.FName + '.SNP') );
        RESET  (SnapF, 4);
        WITH SnapMap^ [S] DO
          BEGIN
            Seek (SnapF, Offset+0);
            IF (Yqty <= Sets.QuickMax)
               THEN QuickY := Yqty
               ELSE QuickY := Sets.QuickMax;
            IF (Xqty <= Sets.QuickMax)
               THEN QuickX := Xqty
               ELSE QuickX := Sets.QuickMax;

            Xratio := (Xqty-1) / (QuickX-1);
            Yratio := (Yqty-1) / (QuickY-1);

            IF (Sets.MapOn)
               THEN BEGIN
                      {Read an averaged/mapped value into the snapshot array}
                      {Ensure first and last values are used unchanged}
                      ix := 0;
                      FOR Xpos := 1 TO QuickX DO
                        IF (NOT(Keypressed)) THEN
                           BEGIN
                             {Init}
                             FOR Ypos := 1 TO QuickY DO
                               BEGIN
                                 SnapShot^ [Xpos, Ypos] := 0;
                                 Count [Ypos] := 0;
                               END;

                             {Accumulate X rows}
                             REPEAT
                               BlockRead (SnapF, SnapVal, Yqty+2); {1 extra 4-byte variable on either side}
                               iy := 0;
                               FOR Ypos := 1 TO QuickY DO
                                 {Accumulate Y values}
                                 REPEAT
                                   Inc (iy);
                                   Inc (Count [Ypos]);
                                   SnapShot^ [Xpos, Ypos] := SnapShot^ [Xpos, Ypos] + SnapVal [iy];
                                 UNTIL (iy >= ROUND((Ypos-1)*Yratio+1) );
                               Inc (ix);
                             UNTIL (ix >= ROUND((Xpos-1)*Xratio+1) );
                             FOR Ypos := 1 TO QuickY DO
                               SnapShot^ [Xpos, Ypos] := SnapShot^ [Xpos, Ypos] / Count [Ypos];
                           END; {For Xpos}
                       END {if MapOn}

               ELSE BEGIN
                      {Read only specific values of the snapshot into the array}
                      {Ensure first and last values are used}
                      ix := 0;
                      FOR Xpos := 1 TO QuickX DO
                        IF (NOT(Keypressed)) THEN
                           BEGIN
                             NextX := ROUND ((Xpos-1)*Xratio+1);
                             {Accumulate X rows}
                             REPEAT
                               Inc (ix);
                               BlockRead (SnapF, SnapVal, Yqty+2); {1 extra 4-byte variable on either side}
                               IF (ix = NextX)
                                  THEN FOR Ypos := 1 TO QuickY DO
                                         SnapShot^ [Xpos, Ypos] := SnapVal [ ROUND((Ypos-1)*Yratio+1) ];
                             UNTIL (ix >= NextX);
                           END; {For Xpos}
                      END;

             CurSnapRec.Xqty   := Xqty;   CurSnapRec.Yqty    := Yqty;
             CurSnapRec.QuickX := QuickX; CurSnapRec.QuickY  := QuickY;
             CurSnapRec.MaxVal := MaxVal; CurSnapRec.MinVal  := MinVal;
             CurSnapRec.Time   := Time;   CurSnapRec.SnapVar := SnapVar; CurSnapRec.Sfunc := Sfunc;
             CurSnapRec.Ax1    := Ax1;    CurSnapRec.Ax2     := Ax2;
             CurSnapRec.Snp    := S;      CurSnapRec.Vpos    := GetVarPos (SnapVar, Ax1, Ax2);

          END; {With SnapMap^[S]}
        CLOSE (SnapF);
      END; {DI_ReadSnap}


    PROCEDURE DI_ExtractSnap;
      CONST
        DumInt : LongInt = 4;
        PlType : LongInt = 0;
      VAR
        SnapF, MxSnapF, MxMapF  : File;
        MxSnap, SnapVal : ARRAY [0..1601] OF Single;
        MxMap           : SnapMapRec;
        last, mqty      : Word;
        S, SS           : Word;
        xx, yy          : Word;
        Pos             : LongInt;
        done, curr      : SET OF Byte;

      BEGIN {DI_ExtractSnap}
        {Open files, also new files *.mxm and *.mxs}
        ASSIGN (SnapF,       (Sets.Drive + ':' + Sets.Dir + '\' + Sets.FName + '.SNP') );
        RESET  (SnapF, 4);
        ASSIGN (MxMapF,      (Sets.Drive + ':' + Sets.Dir + '\' + Sets.FName + '.MXM') );
        REWRITE  (MxMapF, 1);
        ASSIGN (MxSnapF,     (Sets.Drive + ':' + Sets.Dir + '\' + Sets.FName + '.MXS') );
        REWRITE  (MxSnapF, 4);

        {inits}
        last:=0; done:=[]; curr:=[]; mqty:=0;

        {loop through and select a snapshot}
        FOR S:=1 TO QtySnap DO
          IF ( ([SnapMap^[S].Ax1,SnapMap^[S].Ax2]*['T','t']=[]) AND
               NOT(S IN done) ) THEN
             BEGIN
               Inc(mqty);
               MxMap := SnapMap^[S];
               last := S; curr:=[S];
               WITH MxMap DO
                 BEGIN
                   {create set of same type snapshots}
                   FOR SS:=S+1 TO QtySnap DO
                       IF ( ([Ax1, Ax2]*['T','t']=[]) AND NOT(SS IN done)) THEN
                          IF ( (SnapMap^[SS].SnapVar=SnapVar) AND
                               (SnapMap^[SS].Sfunc=Sfunc) AND
                               (SnapMap^[SS].i1=i1) AND (SnapMap^[SS].i2=i2) AND
                               (SnapMap^[SS].j1=j1) AND (SnapMap^[SS].j2=j2) AND
                               (SnapMap^[SS].k1=k1) AND (SnapMap^[SS].k2=k2) AND
                               (SnapMap^[SS].Xqty=Xqty) AND (SnapMap^[SS].Yqty=Yqty)  ) THEN
                             BEGIN
                               Curr := Curr+[SS];
                               IF (SnapMap^[SS].time > time)
                                  THEN time:=SnapMap^[SS].time;
                             END; {if,for}

                   {loop through each snapshot, and for each line extract the
                    maximum, then write to file}
                   FOR xx := 1 TO Xqty DO
                     BEGIN
                       Pos := (xx-1)*(Yqty+2);
                       FOR yy := 1 TO Yqty DO
                         MxSnap[yy] := -largesingle;
                       FOR SS:=S TO QtySnap DO
                         IF (SS IN Curr) THEN
                            BEGIN
                              Seek (SnapF, SnapMap^[SS].Offset+Pos);
                              BlockRead (SnapF, SnapVal, Yqty+2); {2 extra 4-byte variables ..}
                              FOR yy := 1 TO Yqty DO
                                IF (SnapVal[yy] > MxSnap[yy])
                                   THEN MxSnap[yy] := SnapVal[yy];
                            END; {if,for}
                       {write to file}
                       BlockWrite (MxSnapF, MxSnap, Yqty+2);
                       FOR yy := 1 TO Yqty DO
                         BEGIN
                           IF (MxSnap[yy] < MinVal) THEN MinVal := MxSnap[yy];
                           IF (MxSnap[yy] > MaxVal) THEN MaxVal := MxSnap[yy];
                         END;
                     END; {for xx}

                   {write map entry}
                   BlockWrite (MxMapF, DumInt, 4);
                   BlockWrite (MxMapF, PlType, 4);
                   BlockWrite (MxMapF, SnapVar, 78);
                   Seek (MxMapF, FilePos(MxMapF)+42);
                   done := done+curr;
                 END; {With}


             END; {If, For}

        CLOSE (MxSnapF);
        CLOSE (MxMapF);
        CLOSE (SnapF);
      END; {DI_ExtractSnap}



    PROCEDURE DI_ReadDump (D: Word);
      VAR
        ix, iy, Xpos, Ypos      : Word;
        Xratio, Yratio          : Single;
        FPos                    : LongInt;
        Vd1, Vd2, s11, s22, s12 : Single;
        Maj, Minr, MaxV         : Single;
        Summ, Diff, Discr       : Single;
        Sqlen                   : Single;
        Dum1                    : Single;
      BEGIN
        {OBSOLETE - but kept in case of memory shortage}
        WITH DumpMap^ [D] DO
          BEGIN
            IF (Yqty <= MaxDumpSz)
               THEN QuickY := Yqty
               ELSE QuickY := MaxDumpSz;
            IF (Xqty <= MaxDumpSz)
               THEN QuickX := Xqty
               ELSE QuickX := MaxDumpSz;
            IF (QuickY > Sets.QuickMax) THEN QuickY := Sets.QuickMax;
            IF (QuickX > Sets.QuickMax) THEN QuickX := Sets.QuickMax;
            Xratio := Xqty / QuickX;
            Yratio := Yqty / QuickY;

            MaxV := 0;
            FOR Xpos := 1 TO QuickX DO
              IF (NOT(Keypressed) OR (Xpos=1)) THEN
                 BEGIN
                   ix := ROUND((Xpos-0.49)*Xratio) - 1;
                   FPos := Offset + 1 + ix * Yqty * Vqty;
                   Seek (DumpFile, FPos);
                   iy := 0;
                   FOR Ypos := 1 TO QuickY DO
                     BEGIN
                       iy := ROUND ((Ypos-0.49)*Yratio);
                       FPos := Offset + 1 + (ix * Yqty + iy-1) * Vqty;
                       Seek (DumpFile, FPos);
                       IF (Vqty = 9)
                          THEN Read (DumpFile, Vd1, Vd2, Dum1, s11, s22, Dum1, s12, Dum1, Dum1)
                          ELSE Read (DumpFile, Vd1, Vd2, s11, s22, s12);
                       CASE Sets.DumpSet OF
                         Stress : BEGIN
                                    Summ  := s11 + s22;
                                    Diff  := s11 - s22;
                                    Discr := SQRT ( SQR(s12) + 0.25 * SQR (Diff) );
                                    Maj   := 0.5 * Summ - Discr;
                                    Minr  := 0.5 * Summ + Discr;
                                    Dump^ [Xpos, Ypos].Major := Maj;
                                    Dump^ [Xpos, Ypos].Minor := Minr;
                                    Dump^ [Xpos, Ypos].Ang   := 0.5 * Arctan0(2*s12, -Diff);
                                    Maj  := ABS(Maj);
                                    Minr := ABS(Minr);
                                    IF (MaxV < Maj)
                                       THEN MaxV := Maj;
                                    IF (MaxV < Minr)
                                       THEN MaxV := Minr;
                                  END;
                         Vvect  : BEGIN
                                    Dump^ [Xpos, Ypos].V[X] := Vd1;
                                    Dump^ [Xpos, Ypos].V[Y] := Vd2;
                                    Sqlen := SQR(Vd1) + SQR(Vd2);
                                    IF (MaxV < Sqlen)
                                       THEN MaxV := Sqlen;
                                  END;
                       END; {Case}
                     END; {For Ypos}
                 END; {For Xpos}
            IF (Sets.DumpSet = Vvect)
               THEN MaxV := SQRT (MaxV);
            MaxVal := MaxV;
          END; {With DumpMap^[D]}
{        Sound (50); Delay (100); Nosound;}
      END; {DI_ReadDump}



    PROCEDURE DI_ReadDump2 (D: Word);
      TYPE
        DumpRec2D    = RECORD
                         Vd1, Vd2, s11, s22, s12 : Single;
                       END;
        DumpRec3D    = RECORD
                         Vd1, Vd2, Vd3, s11, s22, s33, s12, s23, s13 : Single;
                       END;
        DumpLine2D   = ARRAY [1..3200] OF DumpRec2D;
        DumpLine3D   = ARRAY [1..1600] OF DumpRec3D;
      VAR
        ix, Xpos            : Word;
        NextX               : Word;
        Xratio, Yratio      : Single;
        Mx, Mn              : Single;
        DumpF               : File;
        LineSz              : Word;
        DumpVal2D           : ^DumpLine2D;
        DumpVal3D           : ^DumpLine3D;
        Plane               : Word;
        {temporary variables for particular plot types}
          ssThet, ccThet, scThet,
            s2Thet, c2Thet        : Single;
          Thet, Frict             : Single;

      PROCEDURE SetTempVar;
        BEGIN
          Mx := -1e20;  Mn := +1e20;
          CASE Sets.DumpSet OF
            Stress,
             Vvect : Mx := 0;
            ESS    : BEGIN
                       Thet   := Sets.Thet * PI / 180;
                       Frict  := PI * Sets.Frict / 180;
                       Frict  := ABS ( sin(Frict) / cos(Frict) );
                       ssThet := SQR (SIN(Thet));
                       ccThet := SQR (COS(Thet));
                       scThet := SIN(Thet) * COS(Thet);
                       s2Thet := SIN(2*Thet);
                       c2Thet := COS(2*Thet);
                     END;
          END; {Case}
        END; {SetTmpVar}

      PROCEDURE ConvertXLine (Xpos : Word);
        VAR
          iy, Ypos  : Word;
          Summ, Diff, Discr, vmax   : Single; {Temp reqd for princ. stress}
          Sqlen                     : Single; {Temp reqd for vel. vectors}
          s11d, s22d, s12d          : Single; {Temp reqd for ESS}
          pVd1, pVd2, ps11, ps22, ps12 : Single;   {Values in plane of snapshot}
        BEGIN
          WITH DumpMap^ [D] DO
           FOR Ypos := 1 TO QuickY DO
             BEGIN
{               iy := ROUND((Ypos-1)*Yratio+1);}
               iy := ROUND((Ypos-0.49)*Yratio);
              {Use stresses/ vels in plane of snapshot}
               IF (NOT Dim3)
                  THEN WITH DumpVal2D^[iy] DO
                         BEGIN  pVd1:=Vd1; pVd2:=Vd2; ps11:=s11; ps22:=s22; ps12:=s12;  END
                  ELSE WITH DumpVal3D^[iy] DO
                         CASE Plane OF
                           0: BEGIN  pVd1:=Vd1; pVd2:=Vd2; ps11:=s11; ps22:=s22; ps12:=s12;  END; {x-y}
                           1: BEGIN  pVd1:=Vd1; pVd2:=Vd3; ps11:=s11; ps22:=s33; ps12:=s13;  END; {x-z}
                           2: BEGIN  pVd1:=Vd3; pVd2:=Vd2; ps11:=s33; ps22:=s22; ps12:=s23;  END; {z-y}
                         END; {Case}
              {Now calculate for the required plot type}
               CASE Sets.DumpSet OF
                 Stress : BEGIN
                            Summ  := ps11 + ps22;
                            Diff  := ps11 - ps22;
                            Discr := SQRT ( SQR(ps12) + 0.25 * SQR (Diff) );
                            Dump^ [Xpos, Ypos].Major := 0.5 * Summ - Discr;
                            Dump^ [Xpos, Ypos].Minor := 0.5 * Summ + Discr;
                            Dump^ [Xpos, Ypos].Ang   := 0.5 * Arctan0(2*ps12, -Diff);
                            vmax := Max ( ABS(Dump^[Xpos, Ypos].Major), ABS(Dump^[Xpos, Ypos].Minor) );
                            IF (Mx < vmax) THEN Mx := vmax;
                          END;
                 Vvect  : BEGIN
                            Dump^ [Xpos, Ypos].V[X] := pVd1;
                            Dump^ [Xpos, Ypos].V[Y] := pVd2;
                            Sqlen := SQR(pVd1) + SQR(pVd2);
                            IF (Mx < Sqlen) THEN Mx := Sqlen;
                          END;
                 ESS    : BEGIN
                            s11d := ps11*ccThet + 2*ps12*scThet + ps22*ssThet;
                            s22d := ps11*ssThet - 2*ps12*scThet + ps22*ccThet;
                            s12d := 0.5*(ps11-ps22)*s2Thet + ps12*c2Thet;
                            IF (s22d > 0) THEN s22d := 0;
                            Snapshot^ [Xpos, Ypos] := ABS(s12d) - (Frict * ABS(s22d) );
                          END;
                 ISO    : BEGIN
                            Snapshot^ [Xpos, Ypos] := SQRT ( SQR(ps12) + 0.25 * SQR (ps11-ps22) );
                          END;
               END; {Case}
               IF (Sets.DumpSet IN [ESS, ISO])
                  THEN BEGIN
                         IF (Snapshot^ [Xpos,Ypos] > Mx) THEN Mx := Snapshot^ [Xpos,Ypos];
                         IF (Snapshot^ [Xpos,Ypos] < Mn) THEN Mn := Snapshot^ [Xpos,Ypos];
                       END;
             END; {For and With}
        END; {ConvertXline}


      BEGIN {DI_ReadDump2}
        WITH DumpMap^ [D] DO
          IF (Dim3)
             THEN LineSz := Yqty * SizeOf(DumpVal3D)
             ELSE LineSz := Yqty * SizeOf(DumpVal2D);
        IF (MaxAvail < LineSz)
           THEN BEGIN  DI_ReadDump(D); EXIT;  END; {Take the slower approach}

        {Enough memory - so continue}
        WITH DumpMap^ [D] DO
          BEGIN
            {Get memory and position files}
            IF (Dim3)
               THEN GetMem (DumpVal3D,LineSz)
               ELSE GetMem (DumpVal2D,LineSz);
            ASSIGN (DumpF,     (Sets.Drive + ':' + Sets.Dir + '\' + Sets.FName + '.DMP') );
            RESET  (DumpF, 4);
            Seek (DumpF, Offset+1);
            IF (Xqty <= Sets.QuickMax)
               THEN QuickX := Xqty
               ELSE QuickX := Sets.QuickMax;
            IF (Yqty <= Sets.QuickMax)
               THEN QuickY := Yqty
               ELSE QuickY := Sets.QuickMax;
            IF (Sets.DumpSet IN [Stress,Vvect])
               THEN BEGIN
                      IF (QuickX > MaxDumpSz) THEN QuickX := MaxDumpSz;
                      IF (QuickY > MaxDumpSz) THEN QuickY := MaxDumpSz;
                    END;
            Xratio := Xqty / QuickX;
            Yratio := Yqty / QuickY;
            {Determine plane of plot}
            IF ([Ax1,Ax2] <= ['i','I','j','J'])  THEN Plane:=0;
            IF ([Ax1,Ax2] <= ['i','I','k','K'])  THEN Plane:=1;
            IF ([Ax1,Ax2] <= ['k','K','j','J'])  THEN Plane:=2;

            {Read only specific values from the dump into the array}
            {Ensure first and last values are used}
            SetTempVar;  {Set maxs and mins etc}
            ix := 0;
            FOR Xpos := 1 TO QuickX DO
              IF (NOT(Keypressed) OR (Xpos=1)) THEN
                 BEGIN
                   NextX := ROUND((Xpos-0.49)*Xratio);
                   {Accumulate X rows}
                   REPEAT
                     Inc (ix);
                     IF (Dim3)
                        THEN BlockRead (DumpF, DumpVal3D^, Yqty*Vqty)
                        ELSE BlockRead (DumpF, DumpVal2D^, Yqty*Vqty);
                     IF (ix = NextX)
                        THEN ConvertXline (Xpos);
                   UNTIL (ix >= NextX);
                 END; {For Xpos}

            IF (Sets.DumpSet = Vvect) THEN Mx := SQRT (Mx);
            DumpMap^[D].MaxVal := Mx;
            IF ((Sets.DumpSet IN [Stress, Vvect]) AND (Mx=0))
               THEN DumpMap^[D].MaxVal := 1e10;

            CurSnapRec.Xqty   := Xqty;    CurSnapRec.Yqty    := Yqty;
            CurSnapRec.QuickX := QuickX;  CurSnapRec.QuickY  := QuickY;
            CurSnapRec.MaxVal := Mx;      CurSnapRec.MinVal  := Mn;
            CurSnapRec.Time   := Time;    CurSnapRec.SnapVar := 6;
            CurSnapRec.Ax1    := Ax1;     CurSnapRec.Ax2     := Ax2;
            CurSnapRec.Snp    := D;       CurSnapRec.Vpos    := 0;
            CASE Sets.DumpSet OF
              ESS : CurSnapRec.SnapVar := gcnam0+8;
              ISO : CurSnapRec.SnapVar := gcnam0+3;
            END; {Case}

            CLOSE (DumpF);
            IF (Dim3)
               THEN FreeMem(DumpVal3D,LineSz)
               ELSE FreeMem(DumpVal2D,LineSz);
          END; {With DumpMap^[D]}

{        Sound (50); Delay (100); Nosound;}
      END; {DI_ReadDump2}




    PROCEDURE DI_DumpContour (D: Word);
      VAR
        ix, iy, Xpos, Ypos      : Word;
        Xratio, Yratio          : Single;
        FPos                    : LongInt;
        Mx,Mn                   : Single;
        Vd1, Vd2, s11, s22, s12 : Single;
        s11d, s22d, s12d        : Single;
        ssThet, ccThet, scThet,
          s2Thet, c2Thet        : Single;
        Thet, Frict             : Single;
        Dum1                    : Single;
      BEGIN
        Thet   := Sets.Thet * PI / 180;
        Frict  := PI * Sets.Frict / 180;
        Frict  := ABS ( sin(Frict) / cos(Frict) );
        ssThet := SQR (SIN(Thet));
        ccThet := SQR (COS(Thet));
        scThet := SIN(Thet) * COS(Thet);
        s2Thet := SIN(2*Thet);
        c2Thet := COS(2*Thet);
        Mx := -1e20;  Mn := +1e20;

        WITH DumpMap^ [D] DO
          BEGIN
            IF (Yqty <= Sets.QuickMax)
               THEN QuickY := Yqty
               ELSE QuickY := Sets.QuickMax;
            IF (Xqty <= Sets.QuickMax)
               THEN QuickX := Xqty
               ELSE QuickX := Sets.QuickMax;
            Xratio := Xqty / QuickX;
            Yratio := Yqty / QuickY;

            FOR Xpos := 1 TO QuickX DO
              IF (NOT(Keypressed)) THEN
                 BEGIN
                   ix := ROUND((Xpos-0.49)*Xratio) - 1;
                   iy := 0;
                   FOR Ypos := 1 TO QuickY DO
                     BEGIN
                       iy := ROUND ((Ypos-0.49)*Yratio);
                       FPos := Offset + 1 + (ix * Yqty + iy-1) * Vqty;
                       Seek (DumpFile, FPos);
                       IF (Vqty = 9)
                          THEN Read (DumpFile, Vd1, Vd2, Dum1, s11, s22, Dum1, s12, Dum1, Dum1)
                          ELSE Read (DumpFile, Vd1, Vd2, s11, s22, s12);
                       CASE Sets.DumpSet OF
                         ESS : BEGIN
                                 s11d := s11*ccThet + 2*s12*scThet + s22*ssThet;
                                 s22d := s11*ssThet - 2*s12*scThet + s22*ccThet;
                                 s12d := 0.5*(s11-s22)*s2Thet + s12*c2Thet;
                                 IF (s22d > 0)
                                    THEN s22d := 0;
                                 Snapshot^ [Xpos, Ypos] := ABS(s12d) - (Frict * ABS(s22d) );
                               END;
                         ISO : BEGIN
                                 Snapshot^ [Xpos, Ypos] := SQRT ( SQR(s12) + 0.25 * SQR (s11-s22) );
                               END;
                       END; {Case}
                       IF (Snapshot^ [Xpos,Ypos] > Mx)
                          THEN Mx := Snapshot^ [Xpos,Ypos];
                       IF (Snapshot^ [Xpos,Ypos] < Mn)
                          THEN Mn := Snapshot^ [Xpos,Ypos];
                     END; {For Ypos}
                 END; {For Xpos}
            CurSnapRec.Xqty   := Xqty;    CurSnapRec.Yqty    := Yqty;
            CurSnapRec.QuickX := QuickX;  CurSnapRec.QuickY  := QuickY;
            CurSnapRec.MaxVal := Mx;      CurSnapRec.MinVal  := Mn;
            CurSnapRec.Time   := Time;    CurSnapRec.SnapVar := 6;
            CurSnapRec.Ax1    := Ax1;     CurSnapRec.Ax2     := Ax2;
            CurSnapRec.Snp    := D;       CurSnapRec.Vpos    := 0;
            CASE Sets.DumpSet OF
              ESS : CurSnapRec.SnapVar := gcnam0+8;
              ISO : CurSnapRec.SnapVar := gcnam0+3;
            END; {Case}
          END; {With DumpMap^[D]}
      END; {DI_DumpContour}


    PROCEDURE DI_ReadHist (S: Word; H: HistPtr);
      VAR
        HistVal, DumReal : Single;
        i                : Integer;
      BEGIN
        WITH HistMap^ [S] DO
          BEGIN
            Seek (HistFile, Offset+0);
            FOR i := 1 TO Tqty DO
              Read (HistFile, DumReal, H^[i], DumReal);
          END; {With HistMap^[S]}
      END; {DI_ReadHist}


    PROCEDURE DI_ReadGeom (VAR G: Word;  VAR Gnum, gID  : LongInt);
      CONST
        StateID=$01FEDCB1;  BoundID=$01FEDCB2;
      VAR
        GeomVal, DumReal : Single;
        i,ii             : Integer;
        DumInt           : LongInt;
        Tot, lp, Msize   : Word;
        GFpos, IDcheck   : LongInt;
        LayoutSz         : LongInt;
        FStope           : LongInt;

      PROCEDURE GetGeomExtents;
        VAR
          i                : Integer;
          imin, imax,
           jmin, jmax,
            kmin, kmax     : Integer;
        BEGIN
          imin := 20000; imax := 0;
          jmin := 20000; jmax := 0;
          kmin := 20000; kmax := 0;

          FOR i := 1 TO GeomMap^[G].Mattot DO
            WITH Mats[i] DO
              BEGIN
                IF (i1 < imin) THEN imin := i1;  IF (i2 > imax) THEN imax := i2;
                IF (j1 < jmin) THEN jmin := j1;  IF (j2 > jmax) THEN jmax := j2;
                IF (k1 < kmin) THEN kmin := k1;  IF (k2 > kmax) THEN kmax := k2;
              END;

          FOR i := 1 TO GeomMap^[G].Sourcetot DO
            WITH Sources[i] DO
              BEGIN
                IF (i1 < imin) THEN imin := i1;  IF (i2 > imax) THEN imax := i2;
                IF (j1 < jmin) THEN jmin := j1;  IF (j2 > jmax) THEN jmax := j2;
                IF (k1 < kmin) THEN kmin := k1;  IF (k2 > kmax) THEN kmax := k2;
              END;

          FOR i := 1 TO GeomMap^[G].Stopetot DO
            WITH Stopes^[i] DO
              BEGIN
                IF (i1 < imin) THEN imin := i1;  IF (i2 > imax) THEN imax := i2;
                IF (j1 < jmin) THEN jmin := j1;  IF (j2 > jmax) THEN jmax := j2;
                IF (k1 < kmin) THEN kmin := k1;  IF (k2 > kmax) THEN kmax := k2;
              END;

          WITH GeomMap^ [G] DO
            BEGIN
              i1 := imin; i2 := imax;
              j1 := jmin; j2 := jmax;
              k1 := kmin; k2 := kmax;
            END; {With GeomMap^[G]}
        END; {GetGeomExtents}

      PROCEDURE AdjustGeom;
        VAR
          i, s_i1,s_i2,s_j1,s_j2,s_k1,s_k2, s_qty : integer;
        BEGIN
          FOR i := 1 TO GeomMap^[G].Mattot DO
            WITH Mats[i] DO
              BEGIN  inc(i2); inc(j2); IF (GeomMap^[G].Model3D) THEN inc(k2);  END;
          s_i1:=1; s_i2:=1; s_j1:=1; s_j2:=1; s_k1:=1; s_k2:=1;
          FOR i := 1 TO GeomMap^[G].Sourcetot DO
            WITH Sources[i] DO
              BEGIN
                s_i1 := s_i1+i1; s_i2 := s_i2+i2;
                s_j1 := s_j1+j1; s_j2 := s_j2+j2;
                s_k1 := s_k1+k1; s_k2 := s_k2+k2;
              END;
          WITH GeomMap^[G] DO
            BEGIN
              IF (Sourcetot>0) THEN s_qty:=2*Sourcetot ELSE s_qty:=2;
              COG_i := (s_i1+s_i2) DIV s_qty;
              COG_j := (s_j1+s_j2) DIV s_qty;
              COG_k := (s_k1+s_k2) DIV s_qty;
            END;

          FOR i := 1 TO GeomMap^[G].Stopetot DO
            WITH Stopes^[i] DO
              BEGIN
                {IF (i1 <> i2) THEN inc(i2);
                 IF (j1 <> j2) THEN inc(j2);
                 IF (k1 <> k2) THEN inc(k2);}
              END;
        END; {AdjustGeom}


      BEGIN {DI_ReadGeom}
{       If not given, obtain Geom which matches required ID}
        IF (G=0) THEN
           FOR i := 1 TO QtyGeom DO
             IF ((GeomMap^[i].Gnum = Gnum) AND (GeomMap^[i].gID = gID))
                THEN G := i;
        IF (G>QtyGeom) THEN G:=0;
        IF (G=0) THEN EXIT;
        Gnum:=GeomMap^[i].Gnum;
        gID:=GeomMap^[i].gID;

        WITH GeomMap^[G] DO
          BEGIN
            GFpos := Offset+4;
            Seek (GeomFile, GFpos);
            BlockRead (GeomFile, IDCheck, 4);
            WHILE (IDcheck = StateID) DO
              BEGIN  GFpos:=GFpos+128; Seek (GeomFile, GFpos); BlockRead (GeomFile,IDCheck,4);  END;
            WHILE (IDcheck = BoundID) DO
              BEGIN  GFpos:=GFpos+128; Seek (GeomFile, GFpos); BlockRead (GeomFile,IDCheck,4);  END;

            PropPos := GFpos;
            FOR i := 1 TO Proptot DO
              BEGIN
                Seek (GeomFile, GFpos);
                DumInt := FilePos(GeomFile);
                WITH Props[i] DO
                  BlockRead (GeomFile, Bulk, 12);
                GFpos := GFpos+128;
                IF (i = PropTot) THEN GFpos:=GFpos+128*(PropTotT-PropTot);
              END;
            MatPos := GFpos;
            FOR i := 1 TO Mattot DO
              BEGIN
                Seek (GeomFile, GFpos);
                WITH Mats[i] DO
                  BlockRead (GeomFile, Matnum, 28);
                GFpos := GFpos+128;
                IF (i = MatTot) THEN GFpos:=GFpos+128*(MatTotT-MatTot);
              END;
            FOR i := 1 TO SourceTot DO
              BEGIN
                Seek (GeomFile, GFpos);
                WITH Sources[i] DO
                  BlockRead (GeomFile, stype, 28);
                GFpos := GFpos+128;
                IF (i = SourceTot) THEN GFpos:=GFpos+128*(SourceTotT-SourceTot);
              END;


            FStope:= StopeTotT-StopeTot;
            IF (Fstope<0) THEN Fstope:=0;
            FOR i := 1 TO StopeTotT DO
              BEGIN
                ii:=i-Fstope; {Due to reverse order, and possibility of too many cracks}
                Seek (GeomFile, GFpos);
                IF (ii>0) THEN BlockRead (GeomFile, Stopes^[ii].stopID, 32);
                Seek (GeomFile, GFpos+116);
                BlockRead (GeomFile, LayoutSz, 4);
                GFpos := GFpos+128;
                IF ((LayoutSz<=0) OR (LayoutSz>$FFFE) OR (MaxAvail<LayoutSz))
                   THEN BEGIN
                          IF (ii>0) THEN Stopes^[ii].Layout:=Nil;
                          GFpos := GFpos+LayoutSz +( (128-(LayoutSz MOD 128))MOD 128);   {Padded to 128 blocks}
                        END
                   ELSE BEGIN
                          IF (ii>0) THEN BEGIN
                            Seek (GeomFile, GFpos);
                            GetMem(Stopes^[ii].Layout,LayoutSz);
                            Stopes^[ii].Laysz:=LayoutSz;
                            BlockRead (GeomFile,Stopes^[ii].Layout^[1],LayoutSz);
                          END; {if}
                          GFpos := GFpos+LayoutSz +( (128-(LayoutSz MOD 128))MOD 128);   {Padded to 128 blocks}

(*                        {Still need to write .....}
                          {Extract the outline - at moment assume i-k plane}
                          dT1:=i2-i1+1; dT2:=k2-k1+1;
                          cnt := 0;
                          FOR kp := k1 TO k2 DO
                            FOR ip := i1 TO i2 DO
                              BEGIN
                                inc(cnt);
                                ksol:=Layout^[cnt]=0;
                                IF (kp=k1)
                                   THEN kpsol:=true
                                   ELSE kpsol:=Layout^[cnt-dT1]=0;
                                IF (First)
                                   THEN BEGIN
                                          IF (ksol<>kpsol) THEN
                                             BEGIN
                                               np:=np+1
                                               p[np]:=ip;
                                               First:=False;
                                             END;
                                        END
                                   ELSE BEGIN
                                          IF ((ksol=kpsol) OR (ip=i2))
                                            THEN
                                             BEGIN
                                               p[np].p2:=ip;
                                               First:=True;
                                             END;
                                        END;

                              END; {For}  *)

                        END;

                IF (ii>0) THEN
                 IF (NoOutline)
                    THEN Stopes^[ii].Outline := Nil
                    ELSE BEGIN
                           RESET  (OutLineFile);
                           Readln (OutlineFile,Tot); Msize := (4+Tot*2) * 4;
                           GetMem (Stopes^[i].Outline, Msize);
                           WITH Stopes^[ii].Outline^ DO
                             BEGIN
                               Qty := 1; TotPt := Tot; Pos1 := 1; N1 := Tot;
                               FOR lp := 1 TO Tot DO
                                 Readln (OutlineFile, Pt[lp,1], Pt[lp,2]);
                             END;
                         END; {Else}
              END;
            IF (i1 = 0) THEN GetGeomExtents; {b-compat: older versions}
            GeomMap^[G].Model3D := NOT (GeomMap^[G].k2 = GeomMap^[G].k1);
            AdjustGeom;
          END; {With GeomMap^[G]}
      END; {DI_ReadGeom}

    PROCEDURE DI_ClearGeom (G: Word);
      VAR
        i : Integer;
      BEGIN
        IF (G=0) THEN EXIT;
        FOR i := 1 TO GeomMap^[G].StopeTot DO
          IF (Stopes^[i].Layout<>Nil)
             THEN FreeMem (Stopes^[i].Layout, Stopes^[i].Laysz);
        FOR i := 1 TO GeomMap^[G].StopeTot DO
          IF (Stopes^[i].outline<>Nil)
             THEN FreeMem (Stopes^[i].outline, (Stopes^[i].outline^.Qty*2+4)*4 );
{             THEN BEGIN
                    LayoutSz := Stopes^[i].Layout^[0];
                    FreeMem (Stopes^[i].Layout, LayoutSz+1);
                    Stopes^[i].Layout:=Nil
                  END;}
      END; {DI_ClearGeom}


    PROCEDURE DI_CalcTraject;
        {Select trajectories from the available histories}
      CONST
{        tset=[2,7,5,10,35,40,65,70,95,100,122,127,125,130,155,160,185,190,215,220];}
{        tset=[2,7,95,100, 122,127,215,220];}
{        tset=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,34,35,36,46,47,48,58,59,60];}
         tset=[]; {Allows some to be masked out!! by hand}
      VAR
        ih,j  : Integer;
        tr    : Word;
      BEGIN {Calc_Traject}
        FOR j := 1 TO MaxTraj DO
          WITH TrajMap.List[j] DO
            BEGIN  i1:=0; j1:=0; k1:=0;  his[0]:=0; his[1]:=0; his[2]:=0;  END;
        QtyTraj := 0;
        WITH TrajMap DO
          FOR ih := 1 TO QtyHist DO
            WITH HistMap^[ih] DO
{              IF ((Ax1 IN ['T','t']) AND (HistVar IN [1,2,6, 30,31,33])) THEN}
              IF ((Ax1 IN ['T','t']) AND (HistVar IN [1,2,6, 30,31,33])
                   AND (ih<256) AND NOT((ih MOD 256) IN tset)) THEN
                 BEGIN
                   {Initialize times}
                   IF (QtyTraj = 0) THEN
                      BEGIN  map.t1:=t1; map.t2:=t1; map.dt:=dt; map.Tqty:=Tqty;  END;
                   {Search for an existing entry, or add}
                   tr := 0;
                   FOR j := 1 TO QtyTraj DO
                     IF ( (List[j].i1 = i1) AND (List[j].j1 = j1) AND (List[j].k1 = k1) )
                        THEN tr := j;
                   IF ((tr = 0) AND (QtyTraj < MaxTraj))
                      THEN BEGIN
                             Inc(QtyTraj); tr := QtyTraj;
                             List[tr].i1 := i1;  List[tr].j1 := j1; List[tr].k1 := k1;
                           END;
                   IF (tr <> 0)
                      THEN CASE HistVar OF
                             1,30 : List[tr].his[0] := ih;
                             2,31 : List[tr].his[1] := ih;
                             6,33 : List[tr].his[2] := ih;
                           END; {Case}
                 END; {If,With,For,With}
      END; {DI_CalcTraject}


    PROCEDURE DI_ReadTraject (G:integer);
        {Read the histories associated with trajectory points}
      VAR
        i,ix,tr          : Word;
        tpos             : Word;
        t_h, t_tr        : single;
        HistVal, DumReal : single;
        tval, vel, vel_p : single;
        Dist             : single;
      BEGIN
        IF (QtyTraj=0) THEN EXIT;

        WITH TrajMap DO
        WITH map  DO
        WITH Traj DO
          BEGIN
            Nt    := QtyTraj;
            MemSz := MaxAvail;
{            MemSz := 1024;}
            Np    := TRUNC (Min(Tqty, MemSz/4/3/Nt));
            Tstp  := Dt*Tqty/Np;
            MemSz := Np*Nt*4*3;
            GetMem (Data, MemSz);
            MaxV  := 0;

            FOR tr := 1 TO (Np*Nt*3) DO
              Data^[tr] := 0;

            FOR tr := 1 TO Nt DO
            FOR ix := 0 TO 2 DO
              IF NOT(Keypressed) THEN
              WITH List[tr] DO
                BEGIN
                  IF (his[ix] <> 0) THEN Seek (HistFile, HistMap^[his[ix]].Offset+0);
                  WITH GeomMap^[G] DO
                    Dist := sqrt(sqr((COG_i-List[tr].i1)*dx)
                                +sqr((COG_j-List[tr].j1)*dy)+sqr((COG_k-List[tr].k1)*dz)  );
                  IF (Dist<=0) THEN Dist:=1;
                  tval:=0; vel_p:=0;

                  t_h:=0+Dt*0.5; t_tr:=0; tpos:=(tr-1)*3+ix+1;
                  FOR i := 1 TO Tqty DO
                    BEGIN
                      IF (his[ix] = 0)
                         THEN vel := 0
                         ELSE Read (HistFile, DumReal, vel, DumReal);
                      IF (Traj.Dscale) THEN vel:=vel*Dist; {Scale by distance from source}
                      CASE TrType OF
                        0 : tval := tval + vel*Dt;
                        1 : tval := vel;
                        2 : BEGIN  tval := (vel_p - vel)/Dt;  vel_p := vel;  END;
                        ELSE tval := vel;
                      END; {Case}
                      IF (t_tr <= t_h)
                         THEN BEGIN
                                Data^[tpos] := tval;
                                IF (ABS(tval) >= maxV) THEN maxV := ABS(tval);
                                tpos := tpos+(3*Nt);
                                t_tr := t_tr+Tstp;
                              END;
                      t_h := t_h+Dt;
                    END; {For i}

              END; {With, If, For,For}

            NotRead := False;
            IF (Keypressed) THEN DI_ClearTraject;
          END; {With Traj}
      END; {DI_ReadTraject}

    PROCEDURE DI_ClearTraject;
      BEGIN
        WITH Traj DO
          IF NOT(NotRead) THEN
             BEGIN  FreeMem (Data, MemSz); NotRead:=True;  END;
      END; {DI_ClearTraject}




    {Conversions to other formats - e.g. PSS software, PostScript, Metafiles}


    PROCEDURE DI_CvertPSS (Qty : Integer);
      TYPE
        PSS_Rec = RECORD
                    Event            : LongInt;
                    Chan             : Integer;
                    Day, Month, Year : Integer;
                    Hour, Mins       : Integer;
                    Sec              : Single;
                    Wave             : ARRAY [1..2046] OF Integer;
                  END; {PSS_Rec}
      VAR
        i,j     : Integer;
        Pss     : PSS_Rec;
        PssFile : File OF PSS_Rec;
      BEGIN
        ASSIGN (PssFile, Sets.Drive + ':' + Sets.Dir + '\' + Sets.FName + '.PSS' );
        {$I-} {Turn off IO error checking}
          SEEK (PssFile, FILESIZE (PssFile) );
          IF (IOResult <> 0)
             THEN REWRITE (PssFile);
        {$I+} {Turn on IO error checking}
        WITH Pss DO
          BEGIN
            Day   := 1;
            Month := 1;
            Year  := 91;
            Hour  := 12;
            Mins  := 0;
            Sec   := 0;
          END;
        WITH Pss DO
          FOR i := 1 TO Qty DO
            BEGIN
              Event := 1;
              WITH Seism[i].HstMap DO
                Chan  := i MOD 16;
              FOR j := 1 TO 200 DO
                Wave[j] := 0;
              FOR j := 201 TO 1800 DO
                Wave[j] := ROUND (Seism[i].Hist^ [j-200] / Seism[i].HstMap.Vmax * 2047);
              FOR j := 1801 TO 2046 DO
                Wave[j] := 0;
              Write (PssFile, Pss);
            END;
        CLOSE (PssFile);
      END; {DI_CvertPSS}

    PROCEDURE DI_CvertAscii (Qty, ityp : Integer);
      VAR
        i,j       : Integer;
        AsciiFile : Text;
        t,dt      : single;
      BEGIN
        ASSIGN (AsciiFile, Sets.Drive + ':' + Sets.Dir + '\' + Sets.FName + '.ASC' );
        {$I-} {Turn off IO error checking}
          APPEND (AsciiFile);
          IF (IOResult <> 0)
             THEN REWRITE (AsciiFile);
        {$I+} {Turn on IO error checking}
        WITH Seism[1] DO
          dt := (HstMap.Time - HstMap.Tstart) / (HstMap.Tqty-1);
        t := Seism[1].HstMap.Tstart;
        FOR j := 1 TO Seism[1].HstMap.Tqty DO
          BEGIN
            WITH Seism[1] DO
              IF (ityp=0)
                 THEN write (AsciiFile, t:14, '    ', Hist^[j]:14 )
                 ELSE write (AsciiFile, Hist^[j]:14 );
            FOR i := 2 TO Qty DO
              WITH Seism[i] DO
                write (AsciiFile, '  ', Hist^[j]:14 );
            writeln(AsciiFile);
            t := t+dt;
          END;
        CLOSE (AsciiFile);
      END; {DI_CvertAscii}

    PROCEDURE DI_InitVar;
      BEGIN
        New (SnapMap);
        New (DumpMap);
        New (HistMap);
        New (GeomMap);
        New (CrackDMap);
        New (Stopes);
      END; {DI_InitVar}

    PROCEDURE DI_ClearVar;
      BEGIN
        Dispose (Stopes);
        Dispose (CrackDMap);
        Dispose (GeomMap);
        Dispose (HistMap);
        Dispose (DumpMap);
        Dispose (SnapMap);
      END; {DI_ClearVar}

  BEGIN
  END. {Data_rsc}
