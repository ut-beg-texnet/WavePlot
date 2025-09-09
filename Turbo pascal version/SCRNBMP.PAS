(* ScrnBMP
     - Creates a snapshot of the current Dos screen as a bitmap file
     - Got off the internet, sorry! I didn't note the person's name *)

unit ScrnBMP;

interface

Procedure InitGraf2BMP (Image_path: string);
Procedure GrafToBMP16 (Image_path: string); {16 color BMP}

implementation
uses dos,graph;

Type
  ColorType = Record
                blue: Byte;
                Green: Byte;
                red: Byte;
                Reserved: Byte;
              End;

Procedure GETRGB ( Col: Byte; Var RGB: colortype );
  Var
    RGB_: colortype;
  Begin
    Asm
     Mov Ah, $10;
     Mov Al, $15;
     Mov Bl, Col;
     Int $10
     Mov RGB_.Red, Dh;
     Mov RGB_.Green, Ch;
     Mov RGB_.Blue, Cl;
    End;
    RGB:=RGB_;
   End; {GetRGB}

Const BMPnum:Integer=-1;

Procedure InitGraf2BMP (Image_path: string);
  Var X,Y,I : Integer;
      FN : PathStr;
      SR : SearchRec;
      T : Text;
  Begin
    BMPnum := 0;
    FindFirst(Image_path+'SCRX????.BMP',Archive or ReadOnly,SR);
    while DOSerror=0 do
      Begin
        FN := Copy(SR.Name,5,4);
        Val(FN,X,Y);
        if Y=0 then
        if X >= BMPnum then
           BMPnum := X+1;
        FindNext(SR);
      End;
  End;

Procedure GrafToBMP16 (Image_path: string); {16 color BMP}
  Const
    BMPhead:Array[0..53] of Byte=
          ($42,$4D,$16,$6A,$02,$00,0,0,0,0,$76,0,0,
           0,$28,0,0,0, {no graphix}
           $80,$02,0,0, {bmpsize x}
           $E0,$01,0,0, {bmpsize y}
           $01,0, {bits per plane}
           $04,0, {bits per pixel}
           0,0,0,0, {bitompression}
           0,0,0,0, {imsize}
           0,0,0,0, {resolution x}
           0,0,0,0, {resolution y}
           $10,0,0,0, {colors used}
           0,0,0,0 {colorsimportant}
           );
  Var
    written,X,Y : Integer;
    F : File;
    FN : PathStr;
    B : Array[0..800] of Byte;
    iclr:byte;
    bgr:colortype;
  Begin
    Str(BMPnum:4,FN);
    for X := 1 to 3 do
    if FN[X]=' ' then
    FN[X] := '0'; {fill blanks -> '0'}
    Inc(BMPnum);
    FN := Image_path+'SCRX'+FN; {construct filename}
    FN := FN+'.BMP';
    Assign(F,FN);
    Rewrite(F,1);

    bmphead[18]:=lo(getmaxx+1);
    bmphead[19]:=hi(getmaxx+1);
    bmphead[22]:=lo(getmaxy+1);
    bmphead[23]:=hi(getmaxy+1);

    BlockWrite(F,BMPhead,sizeof(BMPhead),written);

    for iclr:=0 to 15 do
      begin
        getrgb(iclr,bgr);
        bgr.blue:=bgr.blue * 4;
        bgr.green:=bgr.green * 4;
        bgr.red:=bgr.red * 4;
        bgr.reserved:=0;
        blockwrite(F,bgr,sizeof(bgr),written);
      end;

    For Y := getmaxy downto 0 do
      Begin
        for X := 0 to getmaxx div 2 do
          B[X] := GetPixel(X*2+1,Y) OR GetPixel(X*2,Y) shl 4 ; {constuct color bytes of 2 pixels each}
        BlockWrite(F,B,(getmaxx+1) div 2,written);
      End;
    Close(F);
  End; {GrafToBMP16}

begin
  InitGraf2BMP('');
end. {ScrnBMP}
